# OneFormer của ActiveSIGMA trước bước dựng 3D của MAV

Luồng đã tích hợp:

RGB (+ depth/intrinsics) → resize/crop bằng preprocessing của MapAnything →
OneFormer Replica (102 lớp) → MapAnything dựng 3D → submap → pose graph / loop closure
của MAV → xuất map RGB và semantic cùng hình học.

OneFormer đọc đúng RGB đã resize/crop của MAV, hoàn tất trước `MapAnything.infer`.
Mỗi nhãn tương ứng một pixel của point map. Nhãn đi qua cùng mask, confidence, giới hạn
khoảng cách và phép lấy mẫu như RGB; điểm được đưa ra thế giới bằng pose đã tối ưu
`G_i @ inverse(L_i)`, không dùng trực tiếp pose wheel odometry để chồng các frame.
RGB gốc vẫn là đầu vào hình ảnh của MapAnything; nhãn được gắn vào hình học, chưa trở
thành tín hiệu conditioning của mạng hay ràng buộc semantic trong pose graph.

## Chạy

Trong môi trường đã cài requirements của MAV và transformers như môi trường thử
OneFormer, chạy từ `map_reconstruction/`. Dùng scene/depth/wheel bag đã chuẩn bị theo
README của MAV. Thêm `--semantic oneformer` vào lệnh MAV hiện có, ví dụ:

```bash
./build_map.sh <scene_dir> <wheel_odom_bag> <mapping_out_dir> <map_bundle_dir> cuda:2 \
  --rig realsense --depth_dir <scene_dir>/depth_cam \
  --semantic oneformer --semantic_device cuda:2
```

Hoặc chạy trực tiếp (không lưu bundle localization):

```bash
python src/ma_slam/run.py \
  --scene <scene_dir> --mode rgb+depth+intr --depth_dir <scene_dir>/depth_cam \
  --out <mapping_out_dir> --device cuda:2 --rig realsense \
  --semantic oneformer --semantic_device cuda:2 \
  --wheel_odom --wheel_bag <wheel_odom_bag> --wheel_input_pose
```

Có thể thêm `--start_frame 7200 --max_frames 3 --no_loop` để thử đoạn nhỏ trong scene
đã trích xuất; số frame dựa trên tên file scene, không phải vị trí mới sau lọc.
Các dấu `<...>` cần thay bằng đường dẫn scene/bag/output của bạn.

## Kết quả

- `combined_pcd.ply`: XYZ + RGB + `class_id`.
- `combined_semantic.ply`: cùng XYZ và `class_id`, màu theo palette OneFormer đã dùng.
- `semantic_map.json`: thông tin map và quy tắc gộp.
- `semantic/classes.json`, `palette.npy`, `checkpoint_load_report.json`, `inference.json`.
- Thêm `--save_semantic_frames` nếu muốn giữ RGB trên grid MAV, `class_ids.npy` và
  `semantic_color.png` riêng từng ảnh. Mặc định không giữ các ảnh riêng.

Model OneFormer chỉ nạp một lần; cache tối đa 32 ảnh trên CPU giúp dùng lại nhãn ở
vùng overlap. Cache kiểm tra nội dung ảnh đã preprocess, không chỉ tên file.
Loop verification không cần semantic và không chạy lại OneFormer.
Nếu bật voxel downsampling, vị trí/màu lấy trung bình, class bỏ phiếu đa số trên
các quan sát còn lại sau lọc/lấy mẫu; bỏ unknown 0 nếu có class biết; hòa lấy ID nhỏ.
Không thay RGB bằng ảnh màu semantic. Không thay thuật toán dựng 3D hay solver của MAV.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Các test kiểm tra semantic chạy trước forward 3D, nhãn giữ đúng sau pose/mask/range/
sampling, gộp voxel và PLY RGB/semantic có cùng tọa độ + class ID.
Một smoke test riêng ở `/mnt/HDD8/Lp_test/mav_oneformer_integration_smoke` dùng ảnh RGB
7200 đã có, preprocessing MapAnything thật và OneFormer thật, không suy luận lại 500 frame.
Xem `validation.json` tại đó cho phạm vi kiểm tra thực tế.

Checkpoint Replica hiện vẫn có hai tham số layernorm bị thiếu như các lần chạy cũ;
chi tiết được lưu trong checkpoint_load_report.json. Kiểm tra nội bộ không xác nhận
độ chính xác semantic trên dữ liệu robot. Trọng số MapAnything/SALAD cần sẵn có hoặc
được tải khi chạy MAV; test tích hợp dùng backbone giả để kiểm tra thứ tự truyền nhãn,
không chứng minh kết quả dựng 3D của MapAnything thật.

## Kết quả chạy toàn luồng thật

Đã chạy thêm OneFormer + MapAnything thật + MAV pose graph trên 3 frame liên tiếp quanh
60 giây, tạo 2 submap overlap 1 frame. Kết quả: 9.479 voxel 5 cm và 3 pose camera;
RGB/semantic có cùng XYZ và class ID. Lần thử không bật wheel odometry và loop closure.
Xem `/mnt/HDD8/Lp_test/mav_oneformer_3frames_test/validation.json`,
`maps_comparison.png`, `semantic_frames.png` và `run.sh`.

## Kết quả full bag theo mẫu 1/50

Đã chạy thành công trên 217/10.823 RGB front: 44 submap, 303.140 voxel 5 cm,
259 wheel factors, 1 loop closure được chấp nhận. 43 submap dùng input wheel pose prior,
1 submap fallback visual do ngưỡng thời gian. Kiểm tra nội bộ đã qua, chưa đánh giá GT.
Kết quả và lệnh tái hiện tại `/mnt/HDD8/Lp_test/mav_full_front_oneformer/`.
