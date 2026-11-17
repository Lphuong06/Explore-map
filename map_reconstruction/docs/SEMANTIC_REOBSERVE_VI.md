# Database từng frame và nhận xét có cần quan sát thêm

> Tài liệu lịch sử của các policy vùng. Kiến trúc mặc định hiện tại dùng world voxel:
> xem [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md). Các ví dụ dưới đây dùng
> bộ vùng cũ khi thêm `--reobserve_association legacy_region`. Output nặng của các
> lần thử cũ đã được dọn; các đường dẫn kết quả cũ không còn đầy đủ artifact.

Bật bằng `--semantic oneformer --semantic_reobserve` trên lệnh MAV hiện có.
Tính năng này giữ semantic riêng trong database; map xuất ra vẫn là hình học/RGB.
Nếu không bật `--semantic_reobserve`, luồng semantic map hiện có giữ nguyên.

Ví dụ từ thư mục `map_reconstruction/`, trong môi trường MAV:

```bash
python src/ma_slam/run.py \
  --scene <scene_dir> --mode rgb+depth+intr --depth_dir <depth_dir> \
  --out <new_output_dir> --device cuda:0 --rig realsense \
  --semantic oneformer --semantic_device cuda:0 --semantic_reobserve
```

Thêm các tham số wheel/stride/submap hiện có nếu cần. Dùng thư mục output mới;
chưa hỗ trợ resume một database đang có dữ liệu. Không ghi đè các kết quả bag cũ.

## Luồng xử lý

1. OneFormer xử lý RGB trên đúng grid MAV trước forward hình học; lưu class ID đầy đủ
   và tối đa 4096 vector xác suất lấy mẫu đều trên grid. Đây là score không âm chuẩn
   hóa tổng bằng 1, không phải confidence đã được calibration. Chưa lưu toàn bộ tensor
   xác suất dense để tránh dung lượng rất lớn.
2. Sau tối ưu pose graph của mỗi submap, đánh giá các frame mới theo thứ tự trong
   submap. Chỉ dùng các frame đã đưa vào database trước đó, không dùng semantic của
   frame tương lai. Tuy nhiên hình học/pose có ngữ cảnh toàn submap (xử lý theo batch),
   nên kết quả chưa có ngay tại thời điểm một ảnh đơn vừa đến camera.
3. Lưu các điểm mẫu trong hệ camera riêng từng frame. Khi so sánh, lấy pose graph mới
   nhất để chuyển sang world; không dùng tọa độ global đã đóng băng trước loop closure.
4. Trích descriptor SALAD và tìm trên toàn database bằng FAISS cosine similarity,
   không lọc frame theo thứ tự thời gian, submap gần đây hay bán kính camera. Mặc
   định lấy top-50 frame và top-10 frame đại diện nhóm, similarity >= 0.3. Vì có
   index đại diện riêng, nhiều ảnh mới gần giống nhau không tự che khuất mọi frame
   đại diện cũ. Vẫn là truy xuất top-k nên không bảo đảm tìm mọi góc nhìn.
5. Xác minh từng ứng viên qua hai bước: bề mặt 3D chồng lấn (nearest-neighbor <=
   0.15 m, >= 30 mẫu và overlap >= 0.2), rồi ORB (FAST threshold 10) mutual ratio-test (0.8) trên đúng
   grid ảnh MAV. Keypoint được nối với mẫu 3D cách <= 6 pixel; cần >= 12 cặp mẫu
   duy nhất có khoảng cách world <= 0.15 m và tỷ lệ inlier >= 0.3. Chỉ các cặp đã
   xác minh bằng hình ảnh và 3D được dùng tính JSD/entropy đối chiếu.
6. Chỉ yêu cầu quan sát thêm khi entropy hoặc JSD vượt ngưỡng. Thiếu đa dạng góc
   nhìn chỉ là thống kê tham khảo, không phải lý do yêu cầu xem thêm.
7. Ghi nhận xét vào SQLite và JSONL, đồng thời in console. Frame overlap chỉ là một
   bằng chứng; bảng memberships vẫn ghi tất cả submap chứa frame đó.

SALAD được dùng cho truy xuất semantic cả khi `--no_loop`; tùy chọn này chỉ tắt
loop closure của MAV. Khi loop closure bật, tái sử dụng descriptor đã tính cho
submap, không chạy SALAD lần hai. ORB lấy từ RGB MAV đã có.

Nhóm được lưu xuyên submap với frame đại diện (anchor). Frame mới chỉ gia nhập một
nhóm khi xác minh trực tiếp phần chung với anchor. Nếu không, tạo nhóm mới chứa
frame hiện tại và các frame được xác minh trực tiếp. Một frame có thể thuộc nhiều
nhóm. Không gộp bắc cầu chỉ vì A trùng B và B trùng C. Các nhóm thể hiện vùng quan
sát chung, chưa phải định danh instance từng vật. Hai mặt đối diện không có phần
chung không được tự kết luận là cùng vật hoặc mâu thuẫn. ORB có thể bỏ sót vùng ít
texture hoặc góc nhìn thay đổi lớn; sai pose/độ sâu có thể ảnh hưởng xác minh. Chưa
có kiểm chứng visibility bằng z-buffer, xử lý vật động hay độ bất định pose.

## Cách đọc kết quả

Mỗi frame mới có `need_more_observation` (boolean), `status`, `reasons`, `entropy`,
`max_pair_jsd`, `independent_views`, `scene_members`, submap/frame/graph key và đường
 dẫn artifact. `scene_group_id` định danh truy vấn; `scene_group_ids` là các nhóm
bền vững được tham gia. `retrieval_candidates` ghi similarity, nguồn truy xuất và
lý do chấp nhận/từ chối ứng viên; `scene_members` chỉ chứa frame đã xác minh.

- `insufficient_evidence`, `false`: chưa có frame overlap hoặc thiếu hình học hợp lệ.
  Chưa thể kết luận; chờ dữ liệu tự nhiên, không tự động yêu cầu quay lại.
- `observe_more`, `true`: entropy >= 0.55 hoặc mean JSD chuẩn hóa của ít nhất một cặp
  frame >= 0.2. JSD nằm trong [0,1].
- `sufficient`, `false`: có overlap được xác minh và không có
  tín hiệu bất định/mâu thuẫn vượt ngưỡng. Đây là đánh giá nhất quán nội bộ, không
  chứng minh nhãn đúng so với ground truth.

Một góc nhìn được đếm là độc lập khi camera cách mỗi góc đã đếm >= 0.2 m **và**
chênh hướng nhìn về tâm vùng >= 15 độ. Thống kê này không ảnh hưởng quyết định.
`min_independent_views` được giữ để đọc cấu hình cũ, không còn là điều kiện ra quyết định.
Không bắt buộc vật phải được nhìn đủ các mặt; phần khuất không tự gây yêu cầu quan sát thêm. Entropy báo cáo là giá trị lớn nhất giữa trung
bình của frame mới và trung bình phần overlap ở từng frame cùng nhóm. Không fusion
bằng chứng thành một semantic map. Mâu thuẫn của bất kỳ cặp đạt ngưỡng đều yêu cầu
xem thêm; chưa có bỏ phiếu loại outlier hoặc tiêu chí hết ngân sách quan sát.

## File đầu ra

- `semantic/observations/*.npz`: class IDs + pixel indices + normalized scores.
- `semantic/inference.json`, `checkpoint_load_report.json`: nguồn model và tình trạng trọng số.
- `reobserve/observations.sqlite`: bảng frames chứa artifact/nhận xét/pose; bảng
  memberships liên kết frame với các submap chứa nó. Bảng `scene_groups` lưu anchor,
  `scene_group_members` lưu thành viên, `scene_edges` lưu bằng chứng cặp frame.
- `reobserve/submap_<id>/frame_<i>.npz`: class IDs, điểm camera, xác suất mẫu đã lọc,
  intrinsics, pixel indices, SALAD descriptor, ORB descriptors, liên kết keypoint–mẫu
  và đường dẫn semantic gốc.
- `reobserve/decisions.jsonl`: một nhận xét cho mỗi frame mới.
- `reobserve/config.json`: các ngưỡng đã dùng.

SQLite pose được cập nhật theo tối ưu mới nhất; các quyết định JSONL/SQLite vẫn là
nhận xét tại thời điểm frame vào database, không được viết lại hồi tố sau loop closure.
Database lưu được trên đĩa; FAISS index và bằng chứng mẫu hiện giữ trong RAM.
Truy xuất cosine xét toàn bộ vector đã index nhưng chỉ chuyển hình học của ứng viên
sang world. Việc đồng bộ pose SQLite vẫn quét các frame cũ mỗi submap; chưa tối ưu
cho phiên rất dài. `max_camera_distance` và `max_neighbors` chỉ giữ tương thích cấu
hình cũ, không còn giới hạn truy xuất semantic.

Có thể ghi đè ngưỡng bằng JSON, ví dụ:

```json
{"match_distance": 0.15, "min_matches": 30, "min_overlap": 0.2,
 "entropy_threshold": 0.55, "jsd_threshold": 0.2,
 "retrieval_topk": 50, "retrieval_anchor_topk": 10, "retrieval_min_similarity": 0.3,
 "min_local_matches": 12, "min_local_inlier_ratio": 0.3}
```

Truyền `--reobserve_config <file.json>`. Các ngưỡng hiện là giá trị khởi đầu, cần hiệu
chỉnh trên bag thật. `max_range` mặc định 10 m là bộ lọc riêng cho đánh giá, không đổi
cấu hình xuất point cloud. Chưa đề xuất NBV hoặc điều khiển robot.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Tests CPU kiểm tra entropy/JSD, phân biệt cảnh khác dù semantic giống nhau, nhất
quán đa góc nhìn, thiếu góc nhìn, pose correction, lưu database và loại frame overlap.
Các test không đánh giá chất lượng OneFormer/MAV trên dữ liệu robot thật.
Thử nghiệm replay SALAD/ORB thật trên frame 0 và 30, dùng lại hình học đã lưu, nằm
tại `/mnt/HDD8/Lp_test/mav_visual_retrieval_smoke_20261031/fast10/result.json`: frame
30 truy xuất frame 0 với similarity 0.9809, xác minh 24 cặp ảnh–3D, cùng nhóm anchor.
Thử nghiệm này chưa chứng minh truy xuất đúng cho mọi vật hoặc góc nhìn thay đổi lớn.

## Đánh giá lại kết quả cũ theo policy v2

Policy `semantic_evidence_v2` chỉ dùng bất định/mâu thuẫn semantic làm lý do yêu cầu
quan sát thêm. Có thể phân loại lại các số đo đã lưu, không cần chạy lại MAV/OneFormer:

```bash
python tools/semantic/rescore_reobserve.py \
  --input <original_reobserve_dir>/decisions.jsonl \
  --config <original_reobserve_dir>/config.json \
  --output <new_output_dir>
```

Thư mục mới chứa JSONL, CSV và summary; database/nhận xét gốc được giữ nguyên.
Phép phân loại lại không thay nhóm hoặc pose tại thời điểm quan sát ban đầu.

Lưu ý: công cụ rescore v2 chỉ đổi luật quyết định trên số đo cũ. Nó không thể
chuyển các nhóm hình học cũ thành nhóm truy xuất ảnh v3; thay đổi bước grouping cần
chạy lại truy xuất/xác minh từ ảnh và bằng chứng đã lưu, hoặc chạy pipeline mới.

## Quyết định vùng kết hợp semantic–geometry (v4)

Pipeline `MaSlam.run` hiện bật bộ quyết định vùng khi dùng `--semantic_reobserve`.
`ObservationDatabase` có thể nhận `region_config={'enable': True}`; các công cụ
replay cũ không truyền cờ này vẫn chỉ kiểm tra policy v2. Tri-state hiện dùng `null`
cho thiếu bằng chứng xác minh, không còn gộp vào `False`.

Mỗi vùng lưu Dirichlet theo các mẫu bề mặt của anchor. Cặp ORB–3D đã xác minh nối
quan sát mới với đúng sample ID, sau đó đánh giá:

- Semantic consistency: JSD với phân bố trước khi cập nhật, theo từng mẫu.
- Semantic evidence: strength loại prior và posterior entropy. Không kiểm tra phải
  nhìn đủ mặt/góc vật. Frame overlap được loại trùng, nhưng các frame liên tiếp gần
  giống nhau vẫn có thể cung cấp bằng chứng tương quan; strength chưa calibration.
- Geometry confidence: MapAnything confidence tương đối, tỷ lệ mẫu hợp lệ và residual
  3D. Đây là proxy độ tin cậy, không phải covariance của pose hoặc độ chính xác GT.

Geometry yếu được báo riêng trước khi khẳng định semantic conflict. `True` yêu cầu
quan sát thêm, `False` có đủ hỗ trợ trên bề mặt đang xét, `null` chưa thể kết luận.
Các region decision chứa ba cấu trúc `semantic_consistency`, `semantic_evidence`,
`geometry_confidence`; frame decision tổng hợp các vùng đang được đánh giá.

Lượt full 10.823 frame, stride 1, với bằng chứng MAV/OneFormer mới và pose graph online:
`/mnt/HDD8/Lp_test/mav_full_semgeo_20261101/README.md`.

## Chỉ đánh giá frame đầu của mỗi submap

Thêm vào lệnh hiện có:

```bash
--semantic oneformer --semantic_reobserve --reobserve_frame_selection first
```

Frame được chọn là index **0** của submap, gồm cả frame overlap nếu nó đứng đầu.
Với `submap_size=5, overlap=1`, các frame đại diện là 0, 5, 10, 15, ... .
OneFormer chỉ suy luận ảnh này; database quyết định chỉ lưu các frame đại diện.
MAV vẫn xử lý đầy đủ frame trong submap để giữ hình học và pose graph. SALAD của
loop closure vẫn theo luồng MAV hiện có; khi không bật loop, bộ quyết định chỉ tính
descriptor cho ảnh được chọn. Nhãn một ảnh không được xuất nhầm thành semantic map
đầy đủ của submap; output model có `semantic_first_frame_labels` thay vì labels dense.

Không resume database `all` thành `first` vì sẽ trộn bằng chứng của hai chế độ. Dùng
thư mục output mới. Các quyết định vẫn theo ba yếu tố v4 và trả True/False/null.
Mức giảm số lần semantic/decision tùy số frame mới trong submap; không đồng nghĩa
MAV tổng thể nhanh lên cùng tỷ lệ. Test replay dùng 100 submap đã commit của job đã
dừng: `/mnt/HDD8/Lp_test/mav_first_frame_submap_20261101/`.

## Corrected shared-surface semantic gate (v5, 2026-11-02)

`semantic_geometry_regions_v5` requires `verified_shared_surface` and comparable
samples before semantic uncertainty, semantic conflict, or insufficient evidence
can request another observation. First observations and unverified associations
return semantic `null` (Unknown), regardless of provisional entropy or strength.
A low geometry score may independently return final `True`; this is explicitly a
geometry request, while semantic remains Unknown. Outputs separate
`semantic_decision`, `geometry_decision`, and the combined final decision at region
and frame level. Missing viewing angles remain excluded from the decision.

A decision-policy rerun for the saved 1,000-frame test is stored separately at
`/mnt/HDD8/Lp_test/mav_test1000_v5_english_20261102/`. This reuses original online
arrival-time evidence and associations, without a new MAV/OneFormer inference or
reaccumulating evidence. Older v4 outputs and reports are historical and unchanged.

## Semantic evidence shortage also requests observation (v6, 2026-11-02)

At the user's revised request, the active policy is now
`semantic_geometry_regions_v6`. Insufficient semantic evidence independently
returns True, including first observations and unverified associations. The
existing evidence threshold (mass >=1.5 on at least 75% of assessed samples) is
unchanged. This does not assert cross-frame semantic conflict.

Cross-frame conflict and uncertainty require verified corresponding samples and
reliable geometry. The component output now distinguishes `comparison_available`
from overall semantic `evaluable`: evidence can be evaluable without a semantic
comparison. Weak geometry independently returns True; both semantic evidence and
geometry reasons are retained when both request observation. An unknown component
does not override another component's positive observation request. Missing
viewing angles remain excluded. Earlier v4/v5 outputs remain historical.

Fresh all-frame inference for IDs 0–999 is isolated at
`/mnt/HDD8/Lp_test/mav_test1000_allframes_v6_en_20261102/`.
