# Kiến trúc hiện tại: MAV + OneFormer + world voxel

Tài liệu này là điểm bắt đầu cho kiến trúc semantic đang dùng. Các mô tả v2–v6
trong `SEMANTIC_REOBSERVE_VI.md` ghi lại quá trình phát triển bộ đánh giá vùng cũ.

## Luồng xử lý

1. MapAnything chuẩn hóa grid RGB; OneFormer dự đoán trên chính grid đó.
2. MapAnything dựng điểm camera và geometry confidence.
3. MAV nối submap, đóng vòng bằng SALAD và tối ưu pose graph.
4. Cùng chỉ số pixel nối semantic score với điểm 3D; pose MAV đưa điểm sang world.
5. Voxel 10 cm tích lũy bằng chứng Dirichlet, tối đa một đóng góp/frame/voxel.
6. JSD, entropy, lượng bằng chứng và geometry confidence quyết định cần quan sát thêm.
7. Khi pose đổi, đóng góp cũ được trừ và lập chỉ mục lại theo voxel/direction mới.

Semantic được lưu riêng trong SQLite/NPZ. SALAD phục vụ loop closure của MAV;
semantic association dùng world voxel, không truy xuất frame bằng ORB/SALAD.
12 direction bin dùng Fibonacci sphere. Thiếu góc nhìn không tự yêu cầu quan sát lại.
Chưa có lập kế hoạch chuyển động/NBV hoặc persistent object ID.

## Tổ chức code

| Nhiệm vụ | File/thư mục |
|---|---|
| CLI và cấu hình chạy | `src/ma_slam/run.py` |
| Điều phối inference, submap, pose graph | `src/ma_slam/solver.py` |
| MapAnything và chuẩn hóa input | `src/model/` |
| OneFormer và pixel alignment | `src/ma_slam/semantic.py` |
| Chọn bộ lưu bằng chứng | `src/ma_slam/observation.py` |
| World voxel, evidence, pose correction, SQLite | `src/ma_slam/voxel_reobserve.py` |
| Geometry metrics dùng chung | `src/ma_slam/region_evidence.py` |
| Loop retrieval | `src/loop/` |
| Kiểm tra CPU | `tests/` |
| Runner/checkpoint/báo cáo v2 | `tools/experiments/voxel_sigma_v2/` |

`reobserve.py` và `visual_scene.py` còn phục vụ chế độ legacy và các kiểm tra tương
thích. `region_evidence.py` còn cung cấp geometry metrics cho voxel; không xóa module
này như một phiên bản cũ độc lập.

## Chạy pipeline chuẩn

Từ `map_reconstruction/`, trong môi trường đã cài phụ thuộc:

```bash
python src/ma_slam/run.py \
  --scene <scene_dir> --mode rgb+depth+intr --depth_dir <depth_dir> \
  --out <new_output_dir> --device cuda:0 \
  --semantic oneformer --semantic_device cuda:1 --semantic_reobserve
```

`--semantic_reobserve` mặc định chọn `world_voxel`. `--reobserve_config` nhận JSON
ghi đè các ngưỡng trong `voxel_reobserve.DEFAULTS`. Để tái hiện luồng vùng cũ, dùng
`--reobserve_association legacy_region` với bộ ngưỡng cũ tương ứng.
Không dùng chung database giữa hai policy. Chạy vào một thư mục output mới.

Runner 1.000 frame v2 có cơ chế resume/checkpoint riêng, đọc bag và wheel input
của workspace. Lệnh cũ `bash mav_test1000_voxel_sigma_v2_en_20261102/run.sh` vẫn hoạt động.
Biến `MAV_WORKSPACE` và `MAV_EXPERIMENT_OUT` cho phép chọn workspace/output của runner.
CLI chuẩn ở trên không bổ sung resume từ checkpoint của runner.

## Kết quả chuẩn được giữ

`experiments/current/mav_test1000_voxel_sigma_v2_en_20261102/` ở gốc workspace:
1.000 frame, 200 submap, 16 loop; 67 quyết định True, 933 False; 24.592 world voxel.
Giữ nguyên SQLite, artifact camera, checkpoint, snapshot source và toàn bộ báo cáo
của lần chạy này. Đây là số đo nhất quán nội bộ, chưa phải đánh giá độ chính xác GT.
