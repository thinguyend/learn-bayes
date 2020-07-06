---
title: "Chapter 2: Small Worlds and Large Worlds"
description: "Chương 2: Thế giới nhỏ và thế giới lớn"
---

> Bài viết dịch bởi người không chuyên, độc giả nào có góp ý xin phản hồi lại.

```python
import arviz as az
import matplotlib.pyplot as plt

from jax import lax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.contrib.autoguide import (AutoContinuousELBO,
                                       AutoLaplaceApproximation)
import numpyro.distributions as dist
from numpyro.infer import SVI
import numpyro.optim as optim

%config InlineBackend.figure_formats = ["svg"]
az.style.use("arviz-darkgrid")
```
- [2.1 Khu vườn có dữ liệu phân nhánh](#2.1)
- [2.2 Xây dựng model](#2.2)
- [2.3 Các thành phần của model](#2.3)
- [2.4 Cho model chạy](#2.4)
- [2.5 Tổng kết](#2.5)
- [Bài tập](https://nbviewer.jupyter.org/github/vuongkienthanh/learn-bayes/blob/master/notebooks/chap2_ex.ipynb)


Cristoforo Colombo (Christopher Columbus) cho thuyền ra khơi và đi theo hướng Tây vào năm 1492, lúc ấy ông nghĩ rằng Trái Đất hình cầu. Ông cũng giống nhưng người được học khác lúc bấy giờ. Nhưng ông khác ở chỗ ông nghĩ rằng chu vi xích đạo chỉ có 30.000 km chứ thực ra là 40.000 km. Đây là một sai lầm ảnh hưởng nghiêm trọng nhất của lịch sử Châu Âu. Nếu Colombo tính được Trái Đất có chu vi xích đạo là 40.000 thì đoàn thuyền của ông đã mang đủ thức ăn và nước uống để hoàn thành cuộc du hành hướng tây đến Châu Á. Nhưng với 30.000 km, Châu Á sẽ nằm ở bờ Tây của California, và thuyền của ông vẫn có thể mang đủ lương thực khi tới đó. Với sự tự tin vào ước lượng của mình, Colombo ra khơi, và cuối cùng đặt chân đến vùng đất Bahamas.

Colombo dự đoán dựa trên suy nghĩ rằng thế giới chỉ nhỏ thôi. Nhưng ông lại sống trong thế giới lớn, nên vài khía cạnh của dự đoán bị sai. Trường hợp của ông là do may mắn. Mô hình thế giới nhỏ của ông sai một cách không lường trước được: Có rất nhiều đất liền trên đường đi của ông. Nếu mô hình của ông sai theo như mong đợi: không có gì ngoài biến giữa Châu Âu và Châu Á, ông và đoàn thuyền đã phải hết lương thực từ lâu trước khi đến được Đông Ấn.

Mô hình thế giới nhỏ và thế giới lớn của Colombo cho thấy một bức tranh giữa model và hiện thực. Tất cả mọi model thống kê đều có hai khung hình giống nhau: thế giới nhỏ của model và thế giới lớn mà chúng ta muốn áp dùng model lên. Di chuyển giữa hai thế giới vẫn còn là một thách thức lớn của model thống kê. Thách thức còn nghiêm trọng hơn khi ta quên mất sự phân biệt này.

**THẾ GIỚI NHỎ** là một thế giới logic tự giới hạn của model. Trong đó, mọi khả năng đều được biết trước. Không có sự bất ngờ, nhưng sự xuất hiện của một đại lục giữa Châu Âu và Châu Á. Trong thế giới nhỏ, ta cần phải xác định được logic của nó, khẳng định nó hoạt động như mong đợi dưới giả định cho trước. Model Bayesian tốt hơn trong trường hợp này, vì nó dùng những lý lẽ để tối ưu hoá: Không một model thay thế nào sử dụng dữ liệu tốt hơn và hỗ trợ dự đoán chính xác hơn, giả định rằng thế giới nhỏ là mô tả chính xác của thế giới lớn.

**THẾ GIỚI LỚN** là phạm vi mà model triển khai trên đó. Trong thế giới lớn, có thể có những sự kiện không được lồng ghép vào thế giới nhỏ. Có thể nói là, model là một thế giới lớn bị thu nhỏ không hoàn toàn, và nó có thể lầm lỗi, ngay cả khi mọi biến cố đều đã được lồng ghép vào. Logic của model trong thế giới nhỏ là không bảo đảm tối ưu trong thế giới lớn. Nhưng nó là một lời chào ấm áp.

Figure 2-1: Hình vẽ địa cầu 1492 của Martin Behaim, được Colombo dùng. Châu Âu nằm ở bên phải. Châu Á ở bên trái. Hòn đảo lớn tên "Cipangu" là Nhất Bản.
![globe](/assets/images/figure 2-1.png)

Chương này sẽ xây dựng model Bayesian. Cách model Bayesian học từ bằng chứng là tối ưu nhất trong thế giới nhỏ. Khi giả định tương đương thực tế, nó có thể hoạt động tốt trong thế giới lớn. Nhưng hiệu năng của thế giới lớn nên được diễn giải hơn là suy luận logic. Qua lại hai thế giới này cho phép cả hai phương pháp hình thức, như suy diễn Bayesian, và phương pháp không hình thức, như peer review, một vai trò không thể bỏ được.

Chương này tập trung vào thế giới nhỏ. Nó giải thích cho thuyết xác suất ở dạng đơn giản nhất: đếm số biến cố có thể xảy ra. Suy diễn Bayesian xuất hiện từ định nghĩa này. Và sau đó chương này sẽ giới thiệu các thành phần của model thống kê Bayesion, một model dùng để học từ dữ liệu. Và chúng ta sẽ làm model sống dậy, để tạo ra các dự đoán.

Tất cả công việc này sẽ là tiền đề cho chương sau, ở đó bạn được học cách tổng quát hoá ước lượng Bayesian, và xem xét những hiệu ứng của thế giới lớn.

> **Nghĩ lại: Nhanh và nhẹ ở thế giới lớn.** Thế giới tự nhiên rất phức tạp, như làm khoa học là để nhắc lại chúng ta. Nhưng mọi thứ từ con bọ, con sóc hay con đười ươi đều có thể thích nghi được. Động vật không phải Bayesian, bởi vì Bayesian là tốn kém và cần phải có model tốt. Thay vì thế, động vật sử dụng cách để học những kỹ năng mới để sử dụng được trong môi trường của chúng, trong quá khứ lẫn hiện tại. Những kỹ năng này làm cho chúng thích nghi tốt hơn, và có thể hiệu quả hơn là một phân tích Bayesian lớn, khi mà cộng thêm chi phí cho thu thập thông tin, xử lý kết quả (và overfitting). Khi bạn biết thông tin nào cần quan tâm hay bỏ qua, việc dùng hoàn toàn Bayesian là lãng phí. Bayesian không phải điều kiện cần hoặc đủ để ra một quyết định đúng đắn, như các loài động vật đã chứng minh. Nhưng với con người, phân tích Bayesian là một phương pháp tổng quát để phát hiện nhưng thông tin quan trọng và xử lý nó một cách logic. Đừng nghĩ nó là phương pháp duy nhất.

## <center>2.1 Khu vườn có dữ liệu phân nhánh</center><a name="2.1"></a>

Phần này ta sẽ suy diễn Bayesian từ cơ bản nhất. Nó chỉ là phép đếm và so sánh các khả năng. Giả sử ta có câu chuyện tương tự như chuyện ngắn của Jorge Luis Borges "Khu vườn có những lối đi phân nhánh". Câu chuyện kể về một người đàn ông đọc một cuốn sách chứa đầy mâu thuẫn. Tại một thời điểm nhất định, các nhân vật phải quyết định giữa các hướng lựa chọn. Nhân vật chính có thể đến nhà một người đàn ông, có thể giết ông ấy, hay uống một cốc trà. Chỉ được chọn một trong các lựa chọn - giết hoặc trà. Nhưng cuốn sách trong truyện của Borges có tất cả các đường đi. Mỗi quyết định phân nhánh thêm thành một khu vườn có lối đi phân nhánh.

Suy diễn Bayesian cũng dùng thiết bị tương tự. Để suy diễn tốt chuyện gì đã xảy ra, ta nên xem xét tất cả các khả năng. Phân tích Bayesian là một khu vường có dữ liệu phân nhánh, ở đó ta thu thập tất cả các dãy sự kiện thay thế. Khi chúng ta học được những gì đã xảy ra, vài dãy sự kiện được loại bỏ. Cuối cùng, những gì còn lại là phù hợp logic với kiến thức của chúng ta.

Cách tiếp cận này cung cấp hạng bậc cho các giả thuyết, hạng bậc này được bảo tồn tối đa, dưới giả định và dữ liệu. Cách tiếp cận này không đảm bảo rằng cho kết quả đúng, trong thế giới lớn. Nhưng nó đảm bảo cho kết quả tốt nhất có thể, trong thế giới nhỏ, được tính ra từ thông tin được đưa vào model.

Hãy xem ví dụ sau.

### 2.1.1 Đếm các khả năng

Giả sử có một túi chứa 4 viên bi, màu xanh và trắng. Ta biết có 4 viên bi, nhưng không biết số lượng bi của mỗi màu. Ta biết có 5 khả năng: ((1) [WWWW], (2) [BWWW], (3) [BBWW], (4) [BBBW], (5) [BBBB]. Đây là những khả năng đúng với thành phần của cái túi. Ta gọi 5 khả năng này là sự phỏng đoán.

Nhiệm vụ của ta là tìm ra phỏng đoán nào là hợp lý nhất, khi có bằng chứng về nội dung của cái túi. Ta có vài bằng chứng: Ta lấy 3 viên bi từ trong túi ra, từng viên một, để lại vào túi sau khi lấy, và lắc đều túi trước khi lấy viên khác. 3 viên bi này có thứ tự là BWB. Đây là dữ liệu.

Ta bắt đầu trồng khu vường và xem cách dùng data để suy diễn những gì trong túi. Bắt đầu bằng dùng một phỏng đoán ban đầu [BWWW], nghĩa là trong túi có 1 bi xanh và 3 bi trắng. Rút một bi đầu tiên, có 4 khả năng xảy ra, mô phỏng như hình sau:

<center><svg width="120" height="80"><circle cx="11.7" cy="67.06" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="38.87" cy="34.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="81.13" cy="34.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="108.3" cy="67.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="50.34" y1="77.41" x2="21.36" y2="69.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="55.77" y1="70.94" x2="43.1" y2="43.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="64.23" y1="70.94" x2="76.9" y2="43.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="69.66" y1="77.41" x2="98.64" y2="69.65" stroke="black" stroke-width="2" stroke-opacity="1" /></svg></center>

Chú ý rằng với 3 bi trắng nhìn như nhau từ góc nhìn của dữ liệu - ta chỉ ghi nhận màu sắc của viên bi - chúng là sự kiện khác nhau. Điều này quan trọng vì có 3 cách lấy bi trắng hơn là bi xanh.

Giờ ta nhìn lại khu vườn khi rút thêm 1 viên bi, nó nở rộng thêm 1 tầng:

<center><svg width="250" height="120"><circle cx="76.7" cy="107.06" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="103.87" cy="74.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="146.13" cy="74.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="173.3" cy="107.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="115.34" y1="117.41" x2="86.36" y2="109.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="120.77" y1="110.94" x2="108.1" y2="83.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="129.23" y1="110.94" x2="141.9" y2="83.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="134.66" y1="117.41" x2="163.64" y2="109.65" stroke="black" stroke-width="2" stroke-opacity="1" /><circle cx="17.4" cy="97.13" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="22.31" cy="80.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="29.74" cy="65.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="39.51" cy="50.77" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="61.91" cy="29.89" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="76.78" cy="21.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="92.84" cy="14.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="109.69" cy="11.07" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="140.31" cy="11.07" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="157.16" cy="14.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="173.22" cy="21.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="188.09" cy="29.89" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="210.49" cy="50.77" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="220.26" cy="65.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="227.69" cy="80.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="232.6" cy="97.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="67.04" y1="104.47" x2="27.19" y2="99.21" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="67.04" y1="104.47" x2="31.64" y2="84.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="67.04" y1="104.47" x2="38.4" y2="70.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="67.04" y1="104.47" x2="47.29" y2="57.07" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="99.64" y1="65.62" x2="67.64" y2="38.08" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="99.64" y1="65.62" x2="81.16" y2="30.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="99.64" y1="65.62" x2="95.76" y2="24.37" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="99.64" y1="65.62" x2="111.08" y2="20.97" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="150.36" y1="65.62" x2="138.92" y2="20.97" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="150.36" y1="65.62" x2="154.24" y2="24.37" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="150.36" y1="65.62" x2="168.84" y2="30.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="150.36" y1="65.62" x2="182.36" y2="38.08" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="182.96" y1="104.47" x2="202.71" y2="57.07" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="182.96" y1="104.47" x2="211.6" y2="70.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="182.96" y1="104.47" x2="218.36" y2="84.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="182.96" y1="104.47" x2="222.81" y2="99.21" stroke="black" stroke-width="2" stroke-opacity="1" /></svg></center>

Có 16 khả năng xảy ra trong khu vườn, với 2 lần rút bi. Vào lượt rút thứ 2 từ túi, mỗi nhánh lại phân ra thành 4 khả năng. Tại sao?

Bởi vì ta tin rằng khi ta lắc túi trước mỗi lượt rút bi, mỗi viên bi đều có xác suất được rút ra như nhau, không liên quan đến bi trước. Tầng thứ 3 cũng tương tự. Có $4^3=64$ khả năng xảy ra.

Figure 2-2: 64 khả năng khi giả định trong túi có 1 xanh 3 trắng.

<center><svg width="520" height="250"><circle cx="211.7" cy="212.06" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="238.87" cy="179.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="281.13" cy="179.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="308.3" cy="212.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="250.34" y1="222.41" x2="221.36" y2="214.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="255.77" y1="215.94" x2="243.1" y2="188.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="264.23" y1="215.94" x2="276.9" y2="188.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="269.66" y1="222.41" x2="298.64" y2="214.65" stroke="black" stroke-width="2" stroke-opacity="1" /><circle cx="152.4" cy="202.13" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="157.31" cy="185.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="164.74" cy="170.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="174.51" cy="155.77" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="196.91" cy="134.89" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="211.78" cy="126.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="227.84" cy="119.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="244.69" cy="116.07" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="275.31" cy="116.07" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="292.16" cy="119.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="308.22" cy="126.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="323.09" cy="134.89" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="345.49" cy="155.77" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="355.26" cy="170.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="362.69" cy="185.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="367.6" cy="202.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="202.04" y1="209.47" x2="162.19" y2="204.21" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="202.04" y1="209.47" x2="166.64" y2="189.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="202.04" y1="209.47" x2="173.4" y2="175.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="202.04" y1="209.47" x2="182.29" y2="162.07" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="234.64" y1="170.62" x2="202.64" y2="143.08" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="234.64" y1="170.62" x2="216.16" y2="135.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="234.64" y1="170.62" x2="230.76" y2="129.37" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="234.64" y1="170.62" x2="246.08" y2="125.97" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="285.36" y1="170.62" x2="273.92" y2="125.97" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="285.36" y1="170.62" x2="289.24" y2="129.37" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="285.36" y1="170.62" x2="303.84" y2="135.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="285.36" y1="170.62" x2="317.36" y2="143.08" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="317.96" y1="209.47" x2="337.71" y2="162.07" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="317.96" y1="209.47" x2="346.6" y2="175.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="317.96" y1="209.47" x2="353.36" y2="189.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="317.96" y1="209.47" x2="357.81" y2="204.21" stroke="black" stroke-width="2" stroke-opacity="1" /><circle cx="56.0" cy="225.0" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="56.19" cy="216.1" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="56.78" cy="207.22" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="57.75" cy="198.37" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="60.1" cy="184.33" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="62.06" cy="175.65" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="64.4" cy="167.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="67.11" cy="158.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="72.22" cy="145.29" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="75.87" cy="137.18" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="79.88" cy="129.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="84.23" cy="121.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="91.88" cy="109.45" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="97.08" cy="102.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="102.59" cy="95.24" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="108.4" cy="88.5" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="118.29" cy="78.25" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="124.83" cy="72.21" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="131.62" cy="66.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="138.66" cy="61.01" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="150.39" cy="52.95" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="158.0" cy="48.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="165.8" cy="44.05" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="173.79" cy="40.11" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="186.89" cy="34.55" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="195.27" cy="31.54" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="203.77" cy="28.9" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="212.38" cy="26.64" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="226.33" cy="23.8" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="235.14" cy="22.52" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="243.99" cy="21.63" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="252.88" cy="21.12" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="267.12" cy="21.12" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="276.01" cy="21.63" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="284.86" cy="22.52" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="293.67" cy="23.8" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="307.62" cy="26.64" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="316.23" cy="28.9" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="324.73" cy="31.54" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="333.11" cy="34.55" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="346.21" cy="40.11" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="354.2" cy="44.05" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="362.0" cy="48.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="369.61" cy="52.95" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="381.34" cy="61.01" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="388.38" cy="66.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="395.17" cy="72.21" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="401.71" cy="78.25" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="411.6" cy="88.5" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="417.41" cy="95.24" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="422.92" cy="102.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="428.12" cy="109.45" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="435.77" cy="121.46" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="440.12" cy="129.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="444.13" cy="137.18" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="447.78" cy="145.29" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="452.89" cy="158.58" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="455.6" cy="167.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="457.94" cy="175.65" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="459.9" cy="184.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="462.25" cy="198.37" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="463.22" cy="207.22" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="463.81" cy="216.1" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="464.0" cy="225.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><line x1="142.62" y1="200.05" x2="60.0" y2="225.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="142.62" y1="200.05" x2="60.19" y2="216.28" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="142.62" y1="200.05" x2="60.76" y2="207.57" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="142.62" y1="200.05" x2="61.71" y2="198.89" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="147.97" y1="182.0" x2="64.02" y2="185.13" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="147.97" y1="182.0" x2="65.94" y2="176.62" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="147.97" y1="182.0" x2="68.24" y2="168.2" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="147.97" y1="182.0" x2="70.9" y2="159.89" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="156.08" y1="165.0" x2="75.9" y2="146.85" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="156.08" y1="165.0" x2="79.48" y2="138.9" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="156.08" y1="165.0" x2="83.41" y2="131.11" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="156.08" y1="165.0" x2="87.67" y2="123.49" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="166.74" y1="149.48" x2="95.17" y2="111.72" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="166.74" y1="149.48" x2="100.27" y2="104.64" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="166.74" y1="149.48" x2="105.68" y2="97.78" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="166.74" y1="149.48" x2="111.37" y2="91.17" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="191.17" y1="126.7" x2="121.07" y2="81.13" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="191.17" y1="126.7" x2="127.48" y2="75.21" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="191.17" y1="126.7" x2="134.14" y2="69.57" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="191.17" y1="126.7" x2="141.04" y2="64.23" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="207.4" y1="117.14" x2="152.54" y2="56.32" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="207.4" y1="117.14" x2="160.0" y2="51.79" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="207.4" y1="117.14" x2="167.65" y2="47.6" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="207.4" y1="117.14" x2="175.48" y2="43.74" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="224.92" y1="110.24" x2="188.33" y2="38.28" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="224.92" y1="110.24" x2="196.54" y2="35.34" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="224.92" y1="110.24" x2="204.87" y2="32.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="224.92" y1="110.24" x2="213.31" y2="30.53" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="243.3" y1="106.17" x2="226.99" y2="27.74" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="243.3" y1="106.17" x2="235.63" y2="26.49" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="243.3" y1="106.17" x2="244.31" y2="25.62" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="243.3" y1="106.17" x2="253.02" y2="25.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="276.7" y1="106.17" x2="266.98" y2="25.12" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="276.7" y1="106.17" x2="275.69" y2="25.62" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="276.7" y1="106.17" x2="284.37" y2="26.49" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="276.7" y1="106.17" x2="293.01" y2="27.74" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="295.08" y1="110.24" x2="306.69" y2="30.53" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="295.08" y1="110.24" x2="315.13" y2="32.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="295.08" y1="110.24" x2="323.46" y2="35.34" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="295.08" y1="110.24" x2="331.67" y2="38.28" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="312.6" y1="117.14" x2="344.52" y2="43.74" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="312.6" y1="117.14" x2="352.35" y2="47.6" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="312.6" y1="117.14" x2="360.0" y2="51.79" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="312.6" y1="117.14" x2="367.46" y2="56.32" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="328.83" y1="126.7" x2="378.96" y2="64.23" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="328.83" y1="126.7" x2="385.86" y2="69.57" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="328.83" y1="126.7" x2="392.52" y2="75.21" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="328.83" y1="126.7" x2="398.93" y2="81.13" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="353.26" y1="149.48" x2="408.63" y2="91.17" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="353.26" y1="149.48" x2="414.32" y2="97.78" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="353.26" y1="149.48" x2="419.73" y2="104.64" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="353.26" y1="149.48" x2="424.83" y2="111.72" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="363.92" y1="165.0" x2="432.33" y2="123.49" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="363.92" y1="165.0" x2="436.59" y2="131.11" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="363.92" y1="165.0" x2="440.52" y2="138.9" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="363.92" y1="165.0" x2="444.1" y2="146.85" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="372.03" y1="182.0" x2="449.1" y2="159.89" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="372.03" y1="182.0" x2="451.76" y2="168.2" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="372.03" y1="182.0" x2="454.06" y2="176.62" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="372.03" y1="182.0" x2="455.98" y2="185.13" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="377.38" y1="200.05" x2="458.29" y2="198.89" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="377.38" y1="200.05" x2="459.24" y2="207.57" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="377.38" y1="200.05" x2="459.81" y2="216.28" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="377.38" y1="200.05" x2="460.0" y2="225.0" stroke="black" stroke-width="2" stroke-opacity="1" /></svg></center>

Vì chúng ta rút bi ra để tạo dữ liệu, một số nhánh có thể bị loại trừ. Lượt rút đầu tiên là B, nên ta loại 3 phân nhánh W. Nếu ta tưởng tượng data như đi theo một con đường trong vườn phân nhánh, thì nó có nghĩa là đi con đường B từ vị trí ban đầu. Lượt rút thứ hai là W, nên ta giữ 3 nhánh tiếp theo. Ta biết data phải đi từ 1 trong 3 nhánh đó, nhưng không biết nhánh nào, vì ta chỉ ghi nhận màu của bi. Lượt rút cuối là B. Mỗi nhánh của tầng giữa phân ra một nhánh mới. Vậy cuối cùng ta có 3 đường đi tất cả, khi ta giả định trong túi có 1B 3 W. Đây là những đường đi phù hợp nhất với logic của giả định và data.

Figure 2-3: Sau khi loại trừ các nhánh không phù hợp với dữ liệu quan sát được, ta chỉ còn 3 trong 64 nhánh.

<center><svg width="520" height="250"><circle cx="211.7" cy="212.06" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="238.87" cy="179.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="281.13" cy="179.68" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="308.3" cy="212.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="250.34" y1="222.41" x2="221.36" y2="214.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="255.77" y1="215.94" x2="243.1" y2="188.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="264.23" y1="215.94" x2="276.9" y2="188.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="269.66" y1="222.41" x2="298.64" y2="214.65" stroke="black" stroke-width="2" stroke-opacity="0.3" /><circle cx="152.4" cy="202.13" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="157.31" cy="185.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="164.74" cy="170.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="174.51" cy="155.77" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="196.91" cy="134.89" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="211.78" cy="126.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="227.84" cy="119.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="244.69" cy="116.07" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="275.31" cy="116.07" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="292.16" cy="119.81" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="308.22" cy="126.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="323.09" cy="134.89" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="345.49" cy="155.77" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="355.26" cy="170.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="362.69" cy="185.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="367.6" cy="202.13" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="202.04" y1="209.47" x2="162.19" y2="204.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="202.04" y1="209.47" x2="166.64" y2="189.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="202.04" y1="209.47" x2="173.4" y2="175.0" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="202.04" y1="209.47" x2="182.29" y2="162.07" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="234.64" y1="170.62" x2="202.64" y2="143.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="234.64" y1="170.62" x2="216.16" y2="135.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="234.64" y1="170.62" x2="230.76" y2="129.37" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="234.64" y1="170.62" x2="246.08" y2="125.97" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="285.36" y1="170.62" x2="273.92" y2="125.97" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="285.36" y1="170.62" x2="289.24" y2="129.37" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="285.36" y1="170.62" x2="303.84" y2="135.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="285.36" y1="170.62" x2="317.36" y2="143.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="317.96" y1="209.47" x2="337.71" y2="162.07" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="317.96" y1="209.47" x2="346.6" y2="175.0" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="317.96" y1="209.47" x2="353.36" y2="189.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="317.96" y1="209.47" x2="357.81" y2="204.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><circle cx="56.0" cy="225.0" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="56.19" cy="216.1" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="56.78" cy="207.22" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="57.75" cy="198.37" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="60.1" cy="184.33" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="62.06" cy="175.65" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="64.4" cy="167.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="67.11" cy="158.58" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="72.22" cy="145.29" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="75.87" cy="137.18" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="79.88" cy="129.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="84.23" cy="121.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="91.88" cy="109.45" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="97.08" cy="102.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="102.59" cy="95.24" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="108.4" cy="88.5" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="118.29" cy="78.25" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="124.83" cy="72.21" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="131.62" cy="66.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="138.66" cy="61.01" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="150.39" cy="52.95" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="158.0" cy="48.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="165.8" cy="44.05" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="173.79" cy="40.11" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="186.89" cy="34.55" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="195.27" cy="31.54" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="203.77" cy="28.9" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="212.38" cy="26.64" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="226.33" cy="23.8" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="235.14" cy="22.52" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="243.99" cy="21.63" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="252.88" cy="21.12" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="267.12" cy="21.12" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="276.01" cy="21.63" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="284.86" cy="22.52" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="293.67" cy="23.8" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="307.62" cy="26.64" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="316.23" cy="28.9" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="324.73" cy="31.54" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="333.11" cy="34.55" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="346.21" cy="40.11" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="354.2" cy="44.05" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="362.0" cy="48.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="369.61" cy="52.95" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="381.34" cy="61.01" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="388.38" cy="66.46" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="395.17" cy="72.21" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="401.71" cy="78.25" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="411.6" cy="88.5" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="417.41" cy="95.24" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="422.92" cy="102.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="428.12" cy="109.45" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="435.77" cy="121.46" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="440.12" cy="129.23" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="444.13" cy="137.18" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="447.78" cy="145.29" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="452.89" cy="158.58" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="455.6" cy="167.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="457.94" cy="175.65" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="459.9" cy="184.33" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="462.25" cy="198.37" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="463.22" cy="207.22" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="463.81" cy="216.1" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="464.0" cy="225.0" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="142.62" y1="200.05" x2="60.0" y2="225.0" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="142.62" y1="200.05" x2="60.19" y2="216.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="142.62" y1="200.05" x2="60.76" y2="207.57" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="142.62" y1="200.05" x2="61.71" y2="198.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="147.97" y1="182.0" x2="64.02" y2="185.13" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="147.97" y1="182.0" x2="65.94" y2="176.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="147.97" y1="182.0" x2="68.24" y2="168.2" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="147.97" y1="182.0" x2="70.9" y2="159.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="156.08" y1="165.0" x2="75.9" y2="146.85" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="156.08" y1="165.0" x2="79.48" y2="138.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="156.08" y1="165.0" x2="83.41" y2="131.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="156.08" y1="165.0" x2="87.67" y2="123.49" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="166.74" y1="149.48" x2="95.17" y2="111.72" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="166.74" y1="149.48" x2="100.27" y2="104.64" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="166.74" y1="149.48" x2="105.68" y2="97.78" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="166.74" y1="149.48" x2="111.37" y2="91.17" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="191.17" y1="126.7" x2="121.07" y2="81.13" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="191.17" y1="126.7" x2="127.48" y2="75.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="191.17" y1="126.7" x2="134.14" y2="69.57" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="191.17" y1="126.7" x2="141.04" y2="64.23" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="207.4" y1="117.14" x2="152.54" y2="56.32" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="207.4" y1="117.14" x2="160.0" y2="51.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="207.4" y1="117.14" x2="167.65" y2="47.6" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="207.4" y1="117.14" x2="175.48" y2="43.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="224.92" y1="110.24" x2="188.33" y2="38.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="224.92" y1="110.24" x2="196.54" y2="35.34" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="224.92" y1="110.24" x2="204.87" y2="32.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="224.92" y1="110.24" x2="213.31" y2="30.53" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="243.3" y1="106.17" x2="226.99" y2="27.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="243.3" y1="106.17" x2="235.63" y2="26.49" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="243.3" y1="106.17" x2="244.31" y2="25.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="243.3" y1="106.17" x2="253.02" y2="25.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="276.7" y1="106.17" x2="266.98" y2="25.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="276.7" y1="106.17" x2="275.69" y2="25.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="276.7" y1="106.17" x2="284.37" y2="26.49" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="276.7" y1="106.17" x2="293.01" y2="27.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="295.08" y1="110.24" x2="306.69" y2="30.53" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="295.08" y1="110.24" x2="315.13" y2="32.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="295.08" y1="110.24" x2="323.46" y2="35.34" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="295.08" y1="110.24" x2="331.67" y2="38.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="312.6" y1="117.14" x2="344.52" y2="43.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="312.6" y1="117.14" x2="352.35" y2="47.6" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="312.6" y1="117.14" x2="360.0" y2="51.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="312.6" y1="117.14" x2="367.46" y2="56.32" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="328.83" y1="126.7" x2="378.96" y2="64.23" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="328.83" y1="126.7" x2="385.86" y2="69.57" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="328.83" y1="126.7" x2="392.52" y2="75.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="328.83" y1="126.7" x2="398.93" y2="81.13" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="353.26" y1="149.48" x2="408.63" y2="91.17" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="353.26" y1="149.48" x2="414.32" y2="97.78" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="353.26" y1="149.48" x2="419.73" y2="104.64" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="353.26" y1="149.48" x2="424.83" y2="111.72" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="363.92" y1="165.0" x2="432.33" y2="123.49" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="363.92" y1="165.0" x2="436.59" y2="131.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="363.92" y1="165.0" x2="440.52" y2="138.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="363.92" y1="165.0" x2="444.1" y2="146.85" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="372.03" y1="182.0" x2="449.1" y2="159.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="372.03" y1="182.0" x2="451.76" y2="168.2" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="372.03" y1="182.0" x2="454.06" y2="176.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="372.03" y1="182.0" x2="455.98" y2="185.13" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="377.38" y1="200.05" x2="458.29" y2="198.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="377.38" y1="200.05" x2="459.24" y2="207.57" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="377.38" y1="200.05" x2="459.81" y2="216.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="377.38" y1="200.05" x2="460.0" y2="225.0" stroke="black" stroke-width="2" stroke-opacity="0.3" /></svg></center>

Hình này mô tả có 3 đường (trong 64 đường) để tạo ra data BWB từ túi chứa [BWWW]. Chúng ta không có cách để chọn ra đường nào trong 3 đường này. Power suy diễn đến từ so sánh phép đếm này với một phép đếm tương tự của một phỏng đoán thành phần trong túi khác. Ví dụ, xem xét phỏng đoán [WWWW], không có đường nào để phỏng đoán này ra được kết quả BWB, bởi vì không có bi xanh nào trong túi. Tương tự phỏng đoán [BBBB] cũng không phù hợp với data. Ta loại trừ 2 phỏng đoán này.

Figure 2-4: Mô tả khu vườn hoàn thiện.

<center><svg width="580" height="580"><circle cx="277.06" cy="241.7" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="254.64" cy="254.64" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="241.7" cy="277.06" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="241.7" cy="302.94" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="254.64" cy="325.36" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="277.06" cy="338.3" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="302.94" cy="338.3" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="325.36" cy="325.36" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="338.3" cy="302.94" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="338.3" cy="277.06" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="325.36" cy="254.64" r="4" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="302.94" cy="241.7" r="4" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="287.41" y1="280.34" x2="279.65" y2="251.36" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="282.93" y1="282.93" x2="261.72" y2="261.72" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="280.34" y1="287.41" x2="251.36" y2="279.65" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="280.34" y1="292.59" x2="251.36" y2="300.35" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="282.93" y1="297.07" x2="261.72" y2="318.28" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="287.41" y1="299.66" x2="279.65" y2="328.64" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="292.59" y1="299.66" x2="300.35" y2="328.64" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="297.07" y1="297.07" x2="318.28" y2="318.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="299.66" y1="292.59" x2="328.64" y2="300.35" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="299.66" y1="287.41" x2="328.64" y2="279.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="297.07" y1="282.93" x2="318.28" y2="261.72" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="292.59" y1="280.34" x2="300.35" y2="251.36" stroke="black" stroke-width="2" stroke-opacity="0.3" /><circle cx="278.5" cy="180.6" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="267.13" cy="182.4" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="256.01" cy="185.38" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="245.26" cy="189.51" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="225.34" cy="201.01" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="216.4" cy="208.25" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="208.25" cy="216.4" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="201.01" cy="225.34" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="189.51" cy="245.26" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="185.38" cy="256.01" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="182.4" cy="267.13" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="180.6" cy="278.5" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="180.6" cy="301.5" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="182.4" cy="312.87" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="185.38" cy="323.99" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="189.51" cy="334.74" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="201.01" cy="354.66" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="208.25" cy="363.6" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="216.4" cy="371.75" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="225.34" cy="378.99" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="245.26" cy="390.49" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="256.01" cy="394.62" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="267.13" cy="397.6" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="278.5" cy="399.4" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="301.5" cy="399.4" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="312.87" cy="397.6" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="323.99" cy="394.62" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="334.74" cy="390.49" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="354.66" cy="378.99" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="363.6" cy="371.75" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="371.75" cy="363.6" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="378.99" cy="354.66" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="390.49" cy="334.74" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="394.62" cy="323.99" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="397.6" cy="312.87" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="399.4" cy="301.5" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="399.4" cy="278.5" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="397.6" cy="267.13" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="394.62" cy="256.01" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="390.49" cy="245.26" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="378.99" cy="225.34" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="371.75" cy="216.4" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="363.6" cy="208.25" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="354.66" cy="201.01" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="1" fill-opacity="1" /><circle cx="334.74" cy="189.51" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="323.99" cy="185.38" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="312.87" cy="182.4" r="3" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="301.5" cy="180.6" r="3" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="274.47" y1="232.04" x2="279.55" y2="190.55" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="274.47" y1="232.04" x2="269.21" y2="192.19" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="274.47" y1="232.04" x2="259.1" y2="194.89" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="274.47" y1="232.04" x2="249.33" y2="198.65" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="247.57" y1="247.57" x2="231.22" y2="209.1" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="247.57" x2="223.09" y2="215.69" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="247.57" x2="215.69" y2="223.09" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="247.57" x2="209.1" y2="231.22" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="274.47" x2="198.65" y2="249.33" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="274.47" x2="194.89" y2="259.1" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="274.47" x2="192.19" y2="269.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="274.47" x2="190.55" y2="279.55" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="305.53" x2="190.55" y2="300.45" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="305.53" x2="192.19" y2="310.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="305.53" x2="194.89" y2="320.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="232.04" y1="305.53" x2="198.65" y2="330.67" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="332.43" x2="209.1" y2="348.78" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="332.43" x2="215.69" y2="356.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="247.57" y1="332.43" x2="223.09" y2="364.31" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="247.57" y1="332.43" x2="231.22" y2="370.9" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="274.47" y1="347.96" x2="249.33" y2="381.35" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="274.47" y1="347.96" x2="259.1" y2="385.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="274.47" y1="347.96" x2="269.21" y2="387.81" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="274.47" y1="347.96" x2="279.55" y2="389.45" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="305.53" y1="347.96" x2="300.45" y2="389.45" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="347.96" x2="310.79" y2="387.81" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="347.96" x2="320.9" y2="385.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="347.96" x2="330.67" y2="381.35" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="332.43" x2="348.78" y2="370.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="332.43" x2="356.91" y2="364.31" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="332.43" x2="364.31" y2="356.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="332.43" x2="370.9" y2="348.78" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="305.53" x2="381.35" y2="330.67" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="305.53" x2="385.11" y2="320.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="305.53" x2="387.81" y2="310.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="305.53" x2="389.45" y2="300.45" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="347.96" y1="274.47" x2="389.45" y2="279.55" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="274.47" x2="387.81" y2="269.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="274.47" x2="385.11" y2="259.1" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="347.96" y1="274.47" x2="381.35" y2="249.33" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="332.43" y1="247.57" x2="370.9" y2="231.22" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="247.57" x2="364.31" y2="223.09" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="247.57" x2="356.91" y2="215.69" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="332.43" y1="247.57" x2="348.78" y2="209.1" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="305.53" y1="232.04" x2="330.67" y2="198.65" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="232.04" x2="320.9" y2="194.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="232.04" x2="310.79" y2="192.19" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="305.53" y1="232.04" x2="300.45" y2="190.55" stroke="black" stroke-width="2" stroke-opacity="0.3" /><circle cx="283.51" cy="42.08" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="277.02" cy="42.34" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="270.54" cy="42.76" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="264.08" cy="43.36" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="251.2" cy="45.05" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="244.81" cy="46.15" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="238.44" cy="47.42" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="232.11" cy="48.85" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="219.56" cy="52.21" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="213.36" cy="54.14" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="207.22" cy="56.22" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="201.12" cy="58.47" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="189.13" cy="63.44" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="183.23" cy="66.16" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="177.41" cy="69.03" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="171.66" cy="72.05" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="160.42" cy="78.55" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="154.93" cy="82.01" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="149.53" cy="85.62" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="144.23" cy="89.36" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="133.93" cy="97.27" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="128.94" cy="101.42" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="124.06" cy="105.7" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="119.29" cy="110.11" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="110.11" cy="119.29" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="105.7" cy="124.06" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="101.42" cy="128.94" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="97.27" cy="133.93" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="89.36" cy="144.23" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="85.62" cy="149.53" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="82.01" cy="154.93" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="78.55" cy="160.42" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="72.05" cy="171.66" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="69.03" cy="177.41" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="66.16" cy="183.23" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="63.44" cy="189.13" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="58.47" cy="201.12" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="56.22" cy="207.22" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="54.14" cy="213.36" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="52.21" cy="219.56" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="48.85" cy="232.11" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="47.42" cy="238.44" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="46.15" cy="244.81" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="45.05" cy="251.2" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="43.36" cy="264.08" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.76" cy="270.54" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.34" cy="277.02" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.08" cy="283.51" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.08" cy="296.49" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.34" cy="302.98" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="42.76" cy="309.46" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="43.36" cy="315.92" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="45.05" cy="328.8" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="46.15" cy="335.19" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="47.42" cy="341.56" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="48.85" cy="347.89" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="52.21" cy="360.44" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="54.14" cy="366.64" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="56.22" cy="372.78" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="58.47" cy="378.88" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="63.44" cy="390.87" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="66.16" cy="396.77" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="69.03" cy="402.59" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="72.05" cy="408.34" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="78.55" cy="419.58" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="82.01" cy="425.07" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="85.62" cy="430.47" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="89.36" cy="435.77" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="97.27" cy="446.07" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="101.42" cy="451.06" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="105.7" cy="455.94" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="110.11" cy="460.71" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="119.29" cy="469.89" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="124.06" cy="474.3" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="128.94" cy="478.58" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="133.93" cy="482.73" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="144.23" cy="490.64" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="149.53" cy="494.38" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="154.93" cy="497.99" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="160.42" cy="501.45" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="171.66" cy="507.95" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="177.41" cy="510.97" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="183.23" cy="513.84" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="189.13" cy="516.56" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="201.12" cy="521.53" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="207.22" cy="523.78" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="213.36" cy="525.86" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="219.56" cy="527.79" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="232.11" cy="531.15" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="238.44" cy="532.58" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="244.81" cy="533.85" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="251.2" cy="534.95" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="264.08" cy="536.64" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="270.54" cy="537.24" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="277.02" cy="537.66" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="283.51" cy="537.92" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="296.49" cy="537.92" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="302.98" cy="537.66" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="309.46" cy="537.24" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="315.92" cy="536.64" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="328.8" cy="534.95" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="335.19" cy="533.85" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="341.56" cy="532.58" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="347.89" cy="531.15" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="360.44" cy="527.79" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="366.64" cy="525.86" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="372.78" cy="523.78" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="378.88" cy="521.53" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="390.87" cy="516.56" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="396.77" cy="513.84" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="402.59" cy="510.97" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="408.34" cy="507.95" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="419.58" cy="501.45" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="425.07" cy="497.99" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="430.47" cy="494.38" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="435.77" cy="490.64" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="446.07" cy="482.73" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="451.06" cy="478.58" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="455.94" cy="474.3" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="460.71" cy="469.89" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="469.89" cy="460.71" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="474.3" cy="455.94" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="478.58" cy="451.06" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="482.73" cy="446.07" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="490.64" cy="435.77" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="494.38" cy="430.47" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="497.99" cy="425.07" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="501.45" cy="419.58" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="507.95" cy="408.34" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="510.97" cy="402.59" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="513.84" cy="396.77" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="516.56" cy="390.87" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="521.53" cy="378.88" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="523.78" cy="372.78" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="525.86" cy="366.64" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="527.79" cy="360.44" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="531.15" cy="347.89" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="532.58" cy="341.56" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="533.85" cy="335.19" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="534.95" cy="328.8" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="536.64" cy="315.92" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="537.24" cy="309.46" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="537.66" cy="302.98" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="537.92" cy="296.49" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="537.92" cy="283.51" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="537.66" cy="277.02" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="537.24" cy="270.54" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="536.64" cy="264.08" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="534.95" cy="251.2" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="533.85" cy="244.81" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="532.58" cy="238.44" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="531.15" cy="232.11" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="527.79" cy="219.56" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="525.86" cy="213.36" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="523.78" cy="207.22" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="521.53" cy="201.12" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="516.56" cy="189.13" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="513.84" cy="183.23" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="510.97" cy="177.41" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="507.95" cy="171.66" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="501.45" cy="160.42" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="497.99" cy="154.93" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="494.38" cy="149.53" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="490.64" cy="144.23" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="482.73" cy="133.93" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="478.58" cy="128.94" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="474.3" cy="124.06" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="469.89" cy="119.29" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="460.71" cy="110.11" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="455.94" cy="105.7" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="451.06" cy="101.42" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="446.07" cy="97.27" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="435.77" cy="89.36" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="430.47" cy="85.62" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="425.07" cy="82.01" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="1" fill-opacity="1" /><circle cx="419.58" cy="78.55" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="408.34" cy="72.05" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="402.59" cy="69.03" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="396.77" cy="66.16" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="390.87" cy="63.44" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="378.88" cy="58.47" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="372.78" cy="56.22" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="366.64" cy="54.14" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="360.44" cy="52.21" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="347.89" cy="48.85" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="341.56" cy="47.42" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="335.19" cy="46.15" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="328.8" cy="45.05" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="315.92" cy="43.36" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="309.46" cy="42.76" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="302.98" cy="42.34" r="2.5" stroke="black" stroke-width="1.5" fill="blue" stroke-opacity="0.3" fill-opacity="0.3" /><circle cx="296.49" cy="42.08" r="2.5" stroke="black" stroke-width="1.5" fill="white" stroke-opacity="0.3" fill-opacity="0.3" /><line x1="277.46" y1="170.66" x2="283.72" y2="50.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="277.46" y1="170.66" x2="277.44" y2="50.33" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="277.46" y1="170.66" x2="271.17" y2="50.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="277.46" y1="170.66" x2="264.91" y2="51.31" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="265.05" y1="172.62" x2="252.46" y2="52.95" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="265.05" y1="172.62" x2="246.26" y2="54.02" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="265.05" y1="172.62" x2="240.1" y2="55.24" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="265.05" y1="172.62" x2="233.97" y2="56.63" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="175.87" x2="221.84" y2="59.88" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="252.92" y1="175.87" x2="215.84" y2="61.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="175.87" x2="209.89" y2="63.77" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="175.87" x2="203.99" y2="65.94" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="180.37" x2="192.38" y2="70.75" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="241.19" y1="180.37" x2="186.68" y2="73.38" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="180.37" x2="181.04" y2="76.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="180.37" x2="175.48" y2="79.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="192.92" x2="164.6" y2="85.37" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="192.92" x2="159.29" y2="88.72" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="192.92" x2="154.06" y2="92.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="192.92" x2="148.93" y2="95.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="200.82" x2="138.96" y2="103.48" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="200.82" x2="134.13" y2="107.5" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="200.82" x2="129.41" y2="111.65" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="200.82" x2="124.79" y2="115.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="209.7" x2="115.91" y2="124.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="209.7" x2="111.65" y2="129.41" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="209.7" x2="107.5" y2="134.13" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="209.7" x2="103.48" y2="138.96" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="219.47" x2="95.84" y2="148.93" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="219.47" x2="92.21" y2="154.06" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="219.47" x2="88.72" y2="159.29" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="219.47" x2="85.37" y2="164.6" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="241.19" x2="79.08" y2="175.48" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="241.19" x2="76.16" y2="181.04" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="241.19" x2="73.38" y2="186.68" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="241.19" x2="70.75" y2="192.38" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="252.92" x2="65.94" y2="203.99" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="252.92" x2="63.77" y2="209.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="252.92" x2="61.75" y2="215.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="252.92" x2="59.88" y2="221.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="265.05" x2="56.63" y2="233.97" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="265.05" x2="55.24" y2="240.1" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="265.05" x2="54.02" y2="246.26" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="265.05" x2="52.95" y2="252.46" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="277.46" x2="51.31" y2="264.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="277.46" x2="50.74" y2="271.17" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="277.46" x2="50.33" y2="277.44" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="277.46" x2="50.08" y2="283.72" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="302.54" x2="50.08" y2="296.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="302.54" x2="50.33" y2="302.56" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="302.54" x2="50.74" y2="308.83" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="170.66" y1="302.54" x2="51.31" y2="315.09" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="314.95" x2="52.95" y2="327.54" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="314.95" x2="54.02" y2="333.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="314.95" x2="55.24" y2="339.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="172.62" y1="314.95" x2="56.63" y2="346.03" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="327.08" x2="59.88" y2="358.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="327.08" x2="61.75" y2="364.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="327.08" x2="63.77" y2="370.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="175.87" y1="327.08" x2="65.94" y2="376.01" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="338.81" x2="70.75" y2="387.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="338.81" x2="73.38" y2="393.32" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="338.81" x2="76.16" y2="398.96" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="180.37" y1="338.81" x2="79.08" y2="404.52" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="360.53" x2="85.37" y2="415.4" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="360.53" x2="88.72" y2="420.71" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="360.53" x2="92.21" y2="425.94" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="192.92" y1="360.53" x2="95.84" y2="431.07" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="370.3" x2="103.48" y2="441.04" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="370.3" x2="107.5" y2="445.87" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="370.3" x2="111.65" y2="450.59" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="200.82" y1="370.3" x2="115.91" y2="455.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="379.18" x2="124.79" y2="464.09" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="209.7" y1="379.18" x2="129.41" y2="468.35" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="209.7" y1="379.18" x2="134.13" y2="472.5" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="209.7" y1="379.18" x2="138.96" y2="476.52" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="387.08" x2="148.93" y2="484.16" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="219.47" y1="387.08" x2="154.06" y2="487.79" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="219.47" y1="387.08" x2="159.29" y2="491.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="219.47" y1="387.08" x2="164.6" y2="494.63" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="399.63" x2="175.48" y2="500.92" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="399.63" x2="181.04" y2="503.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="399.63" x2="186.68" y2="506.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="241.19" y1="399.63" x2="192.38" y2="509.25" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="404.13" x2="203.99" y2="514.06" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="404.13" x2="209.89" y2="516.23" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="404.13" x2="215.84" y2="518.25" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="252.92" y1="404.13" x2="221.84" y2="520.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="265.05" y1="407.38" x2="233.97" y2="523.37" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="265.05" y1="407.38" x2="240.1" y2="524.76" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="265.05" y1="407.38" x2="246.26" y2="525.98" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="265.05" y1="407.38" x2="252.46" y2="527.05" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="277.46" y1="409.34" x2="264.91" y2="528.69" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="277.46" y1="409.34" x2="271.17" y2="529.26" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="277.46" y1="409.34" x2="277.44" y2="529.67" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="277.46" y1="409.34" x2="283.72" y2="529.92" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="409.34" x2="296.28" y2="529.92" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="409.34" x2="302.56" y2="529.67" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="409.34" x2="308.83" y2="529.26" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="409.34" x2="315.09" y2="528.69" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="407.38" x2="327.54" y2="527.05" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="407.38" x2="333.74" y2="525.98" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="407.38" x2="339.9" y2="524.76" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="407.38" x2="346.03" y2="523.37" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="404.13" x2="358.16" y2="520.12" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="404.13" x2="364.16" y2="518.25" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="404.13" x2="370.11" y2="516.23" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="404.13" x2="376.01" y2="514.06" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="399.63" x2="387.62" y2="509.25" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="399.63" x2="393.32" y2="506.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="399.63" x2="398.96" y2="503.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="399.63" x2="404.52" y2="500.92" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="360.53" y1="387.08" x2="415.4" y2="494.63" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="360.53" y1="387.08" x2="420.71" y2="491.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="360.53" y1="387.08" x2="425.94" y2="487.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="360.53" y1="387.08" x2="431.07" y2="484.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="379.18" x2="441.04" y2="476.52" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="379.18" x2="445.87" y2="472.5" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="379.18" x2="450.59" y2="468.35" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="379.18" x2="455.21" y2="464.09" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="370.3" x2="464.09" y2="455.21" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="370.3" x2="468.35" y2="450.59" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="370.3" x2="472.5" y2="445.87" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="370.3" x2="476.52" y2="441.04" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="360.53" x2="484.16" y2="431.07" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="360.53" x2="487.79" y2="425.94" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="360.53" x2="491.28" y2="420.71" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="360.53" x2="494.63" y2="415.4" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="399.63" y1="338.81" x2="500.92" y2="404.52" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="399.63" y1="338.81" x2="503.84" y2="398.96" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="399.63" y1="338.81" x2="506.62" y2="393.32" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="399.63" y1="338.81" x2="509.25" y2="387.62" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="327.08" x2="514.06" y2="376.01" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="327.08" x2="516.23" y2="370.11" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="327.08" x2="518.25" y2="364.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="327.08" x2="520.12" y2="358.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="314.95" x2="523.37" y2="346.03" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="314.95" x2="524.76" y2="339.9" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="314.95" x2="525.98" y2="333.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="314.95" x2="527.05" y2="327.54" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="409.34" y1="302.54" x2="528.69" y2="315.09" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="409.34" y1="302.54" x2="529.26" y2="308.83" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="409.34" y1="302.54" x2="529.67" y2="302.56" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="409.34" y1="302.54" x2="529.92" y2="296.28" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="409.34" y1="277.46" x2="529.92" y2="283.72" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="409.34" y1="277.46" x2="529.67" y2="277.44" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="409.34" y1="277.46" x2="529.26" y2="271.17" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="409.34" y1="277.46" x2="528.69" y2="264.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="265.05" x2="527.05" y2="252.46" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="265.05" x2="525.98" y2="246.26" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="265.05" x2="524.76" y2="240.1" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="407.38" y1="265.05" x2="523.37" y2="233.97" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="252.92" x2="520.12" y2="221.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="252.92" x2="518.25" y2="215.84" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="252.92" x2="516.23" y2="209.89" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="404.13" y1="252.92" x2="514.06" y2="203.99" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="399.63" y1="241.19" x2="509.25" y2="192.38" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="399.63" y1="241.19" x2="506.62" y2="186.68" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="399.63" y1="241.19" x2="503.84" y2="181.04" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="399.63" y1="241.19" x2="500.92" y2="175.48" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="219.47" x2="494.63" y2="164.6" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="219.47" x2="491.28" y2="159.29" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="219.47" x2="487.79" y2="154.06" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="387.08" y1="219.47" x2="484.16" y2="148.93" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="209.7" x2="476.52" y2="138.96" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="209.7" x2="472.5" y2="134.13" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="209.7" x2="468.35" y2="129.41" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="379.18" y1="209.7" x2="464.09" y2="124.79" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="200.82" x2="455.21" y2="115.91" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="200.82" x2="450.59" y2="111.65" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="200.82" x2="445.87" y2="107.5" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="370.3" y1="200.82" x2="441.04" y2="103.48" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="360.53" y1="192.92" x2="431.07" y2="95.84" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="360.53" y1="192.92" x2="425.94" y2="92.21" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="360.53" y1="192.92" x2="420.71" y2="88.72" stroke="black" stroke-width="2" stroke-opacity="1" /><line x1="360.53" y1="192.92" x2="415.4" y2="85.37" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="180.37" x2="404.52" y2="79.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="180.37" x2="398.96" y2="76.16" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="180.37" x2="393.32" y2="73.38" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="338.81" y1="180.37" x2="387.62" y2="70.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="175.87" x2="376.01" y2="65.94" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="175.87" x2="370.11" y2="63.77" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="175.87" x2="364.16" y2="61.75" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="327.08" y1="175.87" x2="358.16" y2="59.88" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="172.62" x2="346.03" y2="56.63" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="172.62" x2="339.9" y2="55.24" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="172.62" x2="333.74" y2="54.02" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="314.95" y1="172.62" x2="327.54" y2="52.95" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="170.66" x2="315.09" y2="51.31" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="170.66" x2="308.83" y2="50.74" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="170.66" x2="302.56" y2="50.33" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="302.54" y1="170.66" x2="296.28" y2="50.08" stroke="black" stroke-width="2" stroke-opacity="0.3" /><line x1="290" y1="290" x2="290.0" y2="0.0" stroke="black" stroke-width="1" stroke-opacity="1" /><line x1="290" y1="290" x2="38.85" y2="435.0" stroke="black" stroke-width="1" stroke-opacity="1" /><line x1="290" y1="290" x2="541.15" y2="435.0" stroke="black" stroke-width="1" stroke-opacity="1" />
</svg></center>

Hình trên cho thấy khu vườn hoàn thiện với những phỏng đoán còn lại [BWWW], [BBWW], [BBBW]. Phần bên trái của hình cho khu vườn giống với những gì ta đã mô tả với phỏng đoán [BWWW]. Phần bên phải và dưới tương ứng với phỏng đoán [BBWW] và [BBBW]. Bây giờ ta đếm tất cả các đường đi phù hợp với data. Bảng tóm tắt như sau:

| Phỏng đoán | Số cách tạo được BWB |
|----------|----------------------------|
|[WWWW]| 0 x 4 x 0 =0 |
|[BWWW]| 1 x 3 x 1 =3 |
|[BBWW]| 2 x 2 x 2 =8 |
|[BBBW]| 3 x 1 x 3 =9 |
|[BBBB]| 4 x 0 x 4 =0 |

Chú ý số cách tạo ra data của từng phỏng đoán, có thể tính bằng cách đếm số đường đi của từng tầng, và sau đó tính tích của chúng. Đây chỉ là một dụng cụ để tính toán, cho kết quả tương tự như figure 2-4, nhưng không cần phải vẽ ra. Tích của các con số đếm cũng là cách đếm tất cả khả năng theo logic. Đến được đây, bạn đã gặp được dạng trình bày cơ bản của suy diễn Bayesian.

Có gì đặc biệt với những kết quả này? Bằng cách so sánh chúng, ta có biện pháp để đánh giá tính logic giữa các phỏng đoán của thành phần trong túi. Nhưng đây chỉ là một phần của biện pháp, bởi vì để so sánh được, trước tiên ta phải quyết định có bao nhiều cách để tạo phỏng đoán. Ta có thể nói không có lý do gì để giả định khác, ta chỉ xem xét mỗi phỏng đoán đều có xác suất như nhau và có thể so sánh chúng trực tiếp. Tuy nhiên, ta vẫn có lý do để giả định khác.

> **Nghĩ lại: chứng minh.** Sử dụng những phép đếm để đo đạc mối tương quan logic này có thể được chứng minh bằng nhiều cách. Một trong những chứng minh logic là: Nếu chúng ta muốn giải thích về tính logic - chứng minh nó đúng hoặc sai -, ta phải tuân thủ theo quy trình này. Có nhiều phương pháp chứng minh cũng đi đến quy trình toán học này. Cho dù bạn có chọn triết lý nào để chứng minh đi chăng nữa, chỉ cần biết nó hoạt động tốt. Chứng minh và triết lý thúc đẩy quy trình, nhưng cái quan trọng nhất là kết quả. Những ứng dụng suy diễn Bayesian được dùng thành công là tất cả minh chứng bạn cần. Những đối thủ của Suy diễn Bayesian thế kỷ 20 đã lập luận rằng Suy diễn Bayesian dễ chứng minh, nhưng rất khó để áp dụng. May mắn thay nó không còn đúng. Cẩn thận rằng đừng nghĩ rằng chỉ có suy diễn Bayesian là đúng còn tiếp cận khác thì không. Golem có rất nhiều loại, và có vài loại rất hũu ích.

### 2.1.2 Sử dụng thông tin prior (tiền nghiệm)

Chúng ta có thể có thông tin prior từ tính hợp lý tương đối của từng phỏng đoán. Thông tin prior này có từ kiến thức về cách mà nội dung của túi được tạo ra. Hoặc cũng có thể từ data trước. Hoặc chúng ta có thể giả dụ rằng đã có thông tin prior, để xây dựng tính bảo tồn trong phân tích. Cho dù từ nguồn nào, sẽ tốt hơn nếu có cách nào để sử dụng thông tin prior. May mắn là ta có một giải pháp: Chỉ cần dùng phép nhân của prior và số đếm mới.

Để hiểu giải pháp này, giả sử ta có mỗi phỏng đoán đều có khả năng xảy ra như nhau. Ta chỉ cần so sánh số đếm các cách đi mà phù hợp với data lấy được. Phép so sánh này cho thấy [BBBW] nhiều khả năng hơn là [BBWW], và cả hai đều có khả năng xảy ra gấp 3 lần so với [BWWW].

Giờ giả sử ta rút thêm 1 viên bi, là bi xanh. Ta có 2 lựa chọn. Bạn có thể làm lại từ đầu, làm một khu vường có 4 tầng để lần dấu những con đường phù hợp với data [BWBB]. Hoặc bạn dùng số đếm cũ trên những phỏng đoán (0, 3, 8, 9, 0) và cập nhật chúng khi có quan sát mới. Hai phương pháp này đều có cùng phép tính toán học, khi và chỉ khi quan sát mới độc lập với những quan sát cũ.

Bạn sẽ làm như sau. Trước tiên đếm số cách để tạo quan sát mới của mỗi phỏng đoán. Sau đó nhân số đó với những con số của từng phỏng đoán.

| Phỏng đoán | Số cách tạo được [B] | Số đếm cũ | Số đếm mới |
|------------|----------------------|-----------|------------|
|[WWWW]| 0 | 0 | 0 x 0 = 0  |
|[BWWW]| 1 | 3 | 3 x 1 = 3  |
|[BBWW]| 2 | 8 | 8 x 2 = 16 |
|[BBBW]| 3 | 9 | 9 x 3 = 27 |
|[BBBB]| 4 | 0 | 0 x 4 = 0  |

Số đếm mới ở cột tay phải tóm tắt những bằng chứng cho mỗi phỏng đoán. Khi một data mới xuất hiện, và data này không phụ thuộc vào quan sát cũ, thì số lượng các cách đi phù hợp logic tại mỗi phỏng đoán có thể tính bằng cách nhân số đếm cũ với số đếm mới.

Cách tiếp cận cập nhật sẽ đảm bảo rằng (1) khi ta có thông tin trước gợi ý có $W_{\text{prior}}$ cách để tạo ra data cũ $D_{\text{prior}}$ của từng phỏng đoán và (2) ta có những quan sát mới $D_{\text{new}}$ có thể được tạo bằng $W_{\text{new}}$ cách của từng phỏng đoán, thì (3) số cách tạo ra từ $D_{\text{prior}}$ và $D_{\text{new}}$ là tích của $W_{\text{prior}} \times W_{\text{new}}$. Ví dụ, trong phỏng đoán [BBWW] có $W_{\text{prior}}=8$ cách để tạo $D_{\text{prior}}=\text{BWB}$. Nó cũng có $W_{\text{new}} = 2 $ cách để tạo quan sát mới $D_{\text{new}} = \text{B}$. Cho nên có 8 x 2 = 16 cách để phỏng đoán tạo được cả $D_{\text{prior}}$ và $D_{\text{new}}$. Tại sao lại phép nhân? Phép nhân là đường tắt để chia ra và đếm tất cả cả đường đi trong khu vườn mà tạo được tất cả các quan sát.

Trong ví dụ này, prior data và data mới là cùng một loại: các viên bi được rút từ trong túi. Nhưng prior data và data mới vẫn có thể là hai loại khác nhau. Giả sử có một người nói rằng bi xanh rất hiếm. Với mỗi túi chứa [BBBW] được tạo ra, họ tạo 2 túi chứa [BBWW] và 3 túi chứa [BWWW]. Họ đảm bảo rằng ít nhất một bi xanh và một bi trắng trong mỗi túi. Ta có thể cập nhật số đếm của ta như sau:

| Phỏng đoán | Số cách prior | Số đếm nhà sản xuất | Số đếm mới |
|------------|----------------------|-----------|------------|
|[WWWW]| 0  | 0 | 0 x 0  = 0  |
|[BWWW]| 3  | 3 | 3 x 3  = 9  |
|[BBWW]| 16 | 2 | 16 x 2 = 32 |
|[BBBW]| 27 | 1 | 27 x 1 = 27 |
|[BBBB]| 0  | 0 | 0 x 4  = 0  |

Và bây giờ phỏng đoán [BBWW] là phù hợp nhất, nhưng chỉ tốt hơn một chút so với [BBBW]. Có tồn tại một ngưỡng sự khác nhau nào không để chúng ta có thể quyết định phỏng đoán nào là phù hợp hơn một cách an toàn? Chương sau sẽ khám phá câu hỏi này.

>**Nghĩ lại: Thiếu hiểu biết.** Ta dùng giả định nào, khi mà chưa có thông tin trước đây về các phỏng đoán? Giải pháp thông dụng nhất là cho khả năng của từng phỏng đoán là như nhau, trước khi có data. Đây được gọi là **Nguyên tắc không phân bì**: Khi không có lý do nào để nói rằng phỏng đoán này phù hợp hơn phỏng đoán kia, xác suất mọi phỏng đoán đều bằng nhau. Thật ngạc nhiên, vấn đề chọn một hình thức "thiếu hiểu biết" này còn phức tạp hơn nhiều. Vấn đề này sẽ tái xuất hiện trong chương sau. Trong các vấn đề được nêu trong sách, nguyên tắc này cho kết quả suy diễn không khác nhiều so với các phương pháp suy diễn phổ biến khác, với đại đa số giả định mọi khả năng đều có trọng số như nhau. Ví dụ, một khoảng tin cậy non-Bayesian cho xác suất của mọi giá trị có thể của một tham số là bằng nhau, mặc dù vài trong số đó là khá phi lý. Nhiều quy trình Bayesian đã tránh khỏi vấn đề này, thông qua các phương pháp xét phạt, v.v... 

### 2.1.3 Từ phép đếm đến xác suất.

Dựa trên nguyên tắc thiếu hiểu biết "trung thực", ta đã nghĩ ra giải pháp đếm và cho ra khu vườn data phân nhánh.

Nhưng các con số đếm rất khí dùng, nên chúng ta luôn luôn phải chuẩn hoá chúng thành xác suất. Tại sao lại khó? Trước tiên, giá trị tương quan là những gì chúng ta cần, con số 3, 8, 9 không chứa thông tin gì về giá trị. Nó cũng có thể là 30, 80, 90. Ý nghĩa là như nhau. Thứ hai, khi mà data tăng dần, số đếm sẽ lớn lên nhanh chóng và khó kiểm soát. Khi có 10 quan sát, đã có tới hơn một triệu khả năng. Ta phải xử lý dữ liệu với hàng ngàn quan sát, và sử dụng chúng số đếm trực tiếp là bất khả thi.

May mắn là ta có thể dùng toán học để giảm nhẹ công việc này. Cụ thể, ta định nghĩa sự hợp lý được cập nhật sau mỗi lần có data mới như sau:

<center>Sự hợp lý [BWWW] sau khi có data<br/>$\propto$<br/>số cách [BWWW] tạo ra BWB<br/>$\times$<br/>phù hợp ban đầu [BWWW]</center>

Dấu $\propto$ nghĩa là *tỉ lệ với*. Ta muốn so sánh sự phù hợp của mỗi cấu trúc túi khác nhau. Tốt nhất là ta định nghĩa *p* là tỉ lệ của bi xanh trong túi. Trong túi [BWWW], *p* = 1/4 = 0.25. $D_{\text{new}}$ = BWB.

Sự phù hợp của *p* sau $D_{\text{new}} \propto$ cách p tạo ra $D_{\text{new}} \times$ phù hợp ban đầu của *p*

Nghĩa là với một giá trị *p* bất kỳ, ta xem xét sự phù hợp của *p* đó theo tỉ lệ với số đếm tất cả các đường đi trong khu vườn data phân nhánh. Mệnh đề này tóm tắt tất cả những gì ta làm ở phần trước.

Sau cùng, ta tính xác suất dựa trên chuẩn hoá con số sự phù hợp của *p* để cho chúng có tổng là 1. Bạn chỉ cần tính tổng các kết quả tích, rồi lấy mỗi kết quả tích chia cho tổng của chúng.

Sự phù hợp của *p* sau $D_{\text{new}}$ = $\frac{ \text{cách p tạo ra } D_{\text{new}  }   \times \text{phù hợp ban đầu của  p}} {\text{tổng các tích}}$

Để hiểu nó rõ hơn, ta cần một ví dụ. Xem lại bảng cũ:

| Phỏng đoán | *p* | Số cách để có data | Xác suất |
|------------|-----|--------------------|------------|
|[WWWW]| 0    | 0 | 0    |
|[BWWW]| 0.25 | 3 | 0.15 |
|[BBWW]| 0.5  | 8 | 0.40 |
|[BBBW]| 0.75 | 9 | 0.45 |
|[BBBB]| 1    | 0 | 0    |

Ta có thể dùng code để tính nhanh các xác suất.

```python
lst = np.array([0, 3, 8, 9, 0])
plausibilities = lst/np.sum(lst)
# DeviceArray([0.  , 0.15, 0.4 , 0.45, 0.  ], dtype=float32)
```

Những "sự phù hợp" này cũng là xác suất - chúng là số thực không âm (0 hoặc dương)và tổng bằng 1. Và những công thức toán học xác suất đều dùng được với những giá trị này. Cụ thể, mỗi phép tính ứng với một công thức tương tự trong thực hành xác suất.
- Tỉ lệ của bi xanh, *p*, gọi là **PARAMETER** (tham số). Đây là một cách để chỉ điểm những khả năng của data.
- Số lượng tương đối các cách mà *p* có thể tạo ra data thường được gọi là **LIKELIHOOD**. Nó được tính bằng xem qua tất cả các khả năng và loại bỏ những khả năng nào không phù hợp với data.
- Sự phù hợp tiền nghiệm của bất kỳ *p* cụ thể gọi là **PRIOR PROBABILITY**.
- Sự phù hợp mới, được cập nhật, của bất kỳ *p* cụ thể gọi là **POSTERIOR PROBABILITTY**.

>**Nghĩ lại: Ngẫu nhiên hoá.** Khi bạn xáo trộn bộ bài hoặc gán đối tượng vào những can thiệp khác nhau bằng cách tung đồng xu, ta nói bộ bài đó hay việc gán can thiệp vào đối tượng được *ngẫu nhiên hoá*. Thế nó có nghĩa là gì? Có nghĩa là chúng ta xử lý thông tin mà không hề biết cấu trúc của nó như thế nào. Xáo trộn bộ bài làm thay đổi tình trạng thông tin, ta không còn biết thứ tự của bộ bài. Tuy nhiê, điệm cộng từ chuyện này là, nếu ta xáo trộn bộ bài đủ nhiều lần để làm mất hẳn những thông tin prior về thứ tự, thì những lá bài sẽ sắp xếp thành một trong những thứ tự với **ENTROPY THÔNG TIN** (information entropy) cao. Ý tưởng Entropy thông tin sẽ ngày càng quan trọng trong những bài tiếp theo.

## <center>2.2 Xây dựng model</center><a name="2.2"></a>

Bằng cách dùng xác suất thay cho số đếm, Suy diễn Bayesian dễ hơn và khó hơn. Trong phần này, chúng ta theo dõi khu vườn có data phân nhánh bằng đơn giản hoá model Bayesian. Ví dụ dưới đây có cấu trúc của một phân tích thống kê kinh điển, và đây sẽ là dạng chúng ta làm quen. Logic cũng giống như khu vườn phân nhánh thôi.

Giả sử ta có một quả hình cầu mô phỏng Trái Đất. Phiên bản này đủ nhỏ để giữ trên tay. Bạn đang thắc mắc là bao nhiêu phần trăm bề mặt là nước. Bạn có kế hoạch như sau: Bạn tung quả cầu lên. Khi bạn bắt nó, bạn sẽ ghi lại phần bế mặt nằm trên lòng bàn tay của bạn là đất liền hay nước. Và bạn lặp lại tung thêm quả cầu nhiều lần. Kế
hoạch này sẽ tạo ra một dãy các kết quả được lấy mẫu từ quả cầu. Chín mẫu ban đầu có thể nhìn như thế này:

<center> W L W W W L W L W </center>

W đại diện cho nước và L đại diện cho đất liền. Trong ví dụ này, ta có 6 W (nước) và 3 L (đất). Ta gọi đây là data.

Để logic hoạt động, ta phải giả định, và giả định này sẽ là nền của model. Thiết kế một model Bayesian gồm 3 bước:

1. Câu chuyện về data: Thúc đẩy model bằng diễn thuyết cách data xuất hiện.
2. Cập nhật: Huấn luyện model bằng data
3. Lượng giá: Mọi model thống kê đều cần sự dẫn dắt ban đầu và cần phải duyệt lại sau khi huấn luyện.


## 2.2.1 Câu chuyện về data

Phân tích dữ liệu Bayesian thường giống như viết một câu chuyện về dữ liệu xuất hiện như thế nào. Câu chuyện có thể là thuần *mô tả*, xác định những mối liên hệ có thể dùng để dự đoán kết quả, với dữ kiện thêm vào. Hau nó có thể là *quan hệ nhân quả*, một lý thuyết về cách mà một số sự kiện tạo ra sự kiện khác. Thông thường, câu chuyện về quan hệ nhân quả cũng có thể là mô tả. Nhưng nhiều câu chuyện mô tả thì khó là diễn giải bằng quan hệ nhân quả. Nhưng mọi câu chuyện đều hoàn chỉnh, có nghĩa là có khả năng xác định một quy trình để tạo data mới. Công việc mô phỏng data mới là một phần rất quan trọng của lượng giá model.

Bạn có thể thúc đẩy câu chuyện data bằng cách giải thích cách data được sinh ra. Điều này nghĩa là mô tả câu chuyện trong thực tế và cũng như quy trình lấy mẫu. Câu chuyện trong trường hợp này đơn thuần là xây dựng lại các mệnh đề của quy trình lấy mẫu:

1. Tỉ lệ thành phần nước trên bề mặt quả cầu là *p*.
2. Một lần tung cầu thì có xác suất là *p* để tạo nước, và 1 - *p* để tạo đất.
3. Các lần tung không phụ thuộc lẫn nhau.

Câu chuyện này được dịch sang một model xác suất. Model dễ xây dựng, vì nó có thể chia ra các thành phần nhỏ hơn. Trước khi gặp các thành phần này, ta sẽ xem qua lược đồ cách model Bayesian hoạt động. 

>**Nghĩ lại: Giá trị của câu chuyện.** Câu chuyện data có giá trị, ngay cả nếu bạn từ bỏ nó nhanh chóng và không bao giờ dùng nó để tạo model hoặc mô phỏng data mới. Đúng vậy, việc từ từ loại bỏ câu chuyện rất quan trọng, vì có rất nhiều câu chuyện tương ứng với cùng một model. Kết quả là, một model làm việc tốt không có nghĩa là phù hợp với câu chuyện data. Nhưng nó vẫn có tác dụng, vì nó là bản nháp của câu chuyện, và sau đó ta sẽ nhận ra những câu hỏi thêm để hoàn thành câu chuyện. Tất cẳ các câu chuyện data thường phải cụ thể hơn là mô tả ngôn từ. Giả thuyết có thể mơ hồ, ví dụ như "trời thường đổ mưa khi nhiệt độ ngày ấm". Khi bạn phải lấy mẫu và đo đạc và tạo mệnh đề nhiệt độ ngày như thế nào thì có mưa, rất nhiều câu chuyện và model có thể phù hợp với giả thuyết mơ hồ như vậy. Giải quyết sự mơ hồ này thường dẫn tới sự nhận ra những câu hỏi quan trọng và tái thiết lập model, trước khi fit data.

### 2.2.2 Cập nhật Bayesian

Vấn đề bây giờ là dùng chứng cứ - một chuỗi sự kiện tung quả cầu - để quyết định tỉ lệ nước trên bề mặt quả cầu. Tỉ lệ này giống như tỉ lệ bi xanh trong túi. Mỗi khả năng của tỉ lệ có phù hợp hơn hoặc kém, tuỳ theo chứng cứ. Một model Bayesian bắt đầu bằng một tập các sự phù hợp ứng với một tập các khả năng. Đây là tiền nghiệm. Sau đó nó cập nhật bằng sự xuất hiện của data, để tạo ra hậu nghiệm. Phương pháp này gọi là Cập nhất Bayesian. 

Ví dụ, model Bayesian khởi đầu bằng cách gán xác suất của *p* bằng nhau với mọi *p*, tỉ lệ nước bề mặt quả cầu.

![](/assets/images/figure 2-5.png)

Nhìn vào hình trên ở góc trên trái, đường ngang nét chấm là khả năng ban đầu của mọi giá trị *p*. Sau khi tung cầu lần 1, "W", model cập nhật khả năng của *p* thành đường nét liền. Khả năng *p*=0 giảm xuống thành chính xác là 0 - tương ứng với "không thể nào". Tại sao? Tại vì ta quan sát được ít nhất một phần nước của quả cầu, và ta biết có nước trên quả cầu. Model tự động thực hiện logic này. Bạn không cần phải dạy nó nhớ hệ quả này. Thuyết xác suất làm giúp bạn, bởi vì nó luôn luôn đếm số cách đi trong khu vườn data.

Tương tự, khả năng của *p*>5 tăng lên. Bởi vì chưa có bằng chứng rằng có đất trên bề mặt quả cầu, nên khả năng đầu tiên được chỉnh và phù hợp với điều này. Chú ý rằng khả năng tương đối là những gì chúng ta quan tâm. Vì chưa có nhiều data, nên sự khác nhau này chưa có lớn. Bằng cách này, bằng chứng hiện tại đã được đưa vào xác suất của mỗi giá trị *p*.

Trong những hình còn lại, những quan sát mới của quả cầu được đưa vào model từng lượt một. Đường nét đứt là đường nét liền của hình cũ, từ trái sang phải, từ trên xuống dưới. Mỗi lần "W" được thấy, đỉnh của đường cong xác suất sẽ đi sang phải, hướng về giá trị *p* lớn hơn. Mỗi lần "L" được thấy, nó hướng ngược lại. Chiều cao lớn nhất của đường cong tăng lên theo mỗi lượt lấy mẫu, nghĩa là ngày càng ít giá trị *p* có khả năng cao khi số lượng mẫu tăng lên. Mỗi lần có quan sát mới, đường cong được cập nhật phù hợp luôn với những quan sát cũ.

Chú ý rằng mỗi xác suất được cập nhật hiện tại sẽ là xác suất ban đầu của lượt quan sát tiếp theo. Mỗi kết luận là điểm khởi đầu của suy diễn tương lai. Tuy nhiên, quy trình này làm ngược lại cũng tốt như tiến tới. Xem vào hình cuối cùng, biết rằng quan sát cuối là W, xác suất có thể bị bù trừ và trở thành đường cong trước. Cho nên data có thể xuất hiện trước hoặc sau, thứ tự nào cũng được. Thông thường, bạn sẽ trình diễn toàn bộ data cùng một lúc cho tiện. Nhưng bạn cũng phải nhận ra rằng đây là một quy trình tự học lặp lại nhiều lần.

>**Nghĩ lại: cỡ mẫu và suy diễn đáng tin cậy.** Bạn thường nghe là có một cỡ mẫu tối thiểu cho một phương pháp thống kê nào đó. Ví dụ, phải có 30 mẫu quan sát mới dùng được phân phối Gaussian. Tại sao? Trong suy diễn non-Bayesian, quy trình thường được chỉnh sửa bởi vì hành vi của model hoạt động ở cỡ mẫu rất lớn, gọi là **hành vi tiệm cận**. Kết quả là, hiệu năng khi cỡ mẫu nhỏ thường phải được đặt nghi vấn.
>Ngược lại, ước lượng Bayesian là an toàn với mọi cỡ mẫu. Điều này không có nghĩa là nhiều data không có ích - dĩ nhiên là có. Ước lượng này cho ta một diễn giải rõ ràng và có căn cứ, với bất kỳ cỡ mẫu nào. Nhưng cái giá cho sức mạnh này là tính phụ thuộc vào prior, ước lượng ban đầu. Khi prior không tốt, kết quả từ suy diễn sẽ dễ gây hiểu nhầm. Không có bữa ăn nào miễn phí, khi bạn học từ thế giới bên ngoài. Một golem Bayesian phải chọn những xác suất ban đầu, và một golem non-Bayesian phải chọn một đại lượng ước lượng. Cả 2 golem đều phải trả giá cho bữa ăn với giả định của chính nó.

### 2.2.3 Lượng giá

Model Bayesian học data theo cách đã được chứng minh là tối ưu, với giả định rằng model mô phỏng chính xác thế giới thực. Có thể nói rằng Bayesian đảm bảo suy luận chính xác trong thế giới nhỏ. Không có cỗ máy nào sử dụng thông tin có sẵn, bắt đầu từ một trạng thái thông tin, có thể hiệu quả hơn Bayesian.

Nhưng đừng có bị cám dỗ. Các phép tính có thể bị lỗi, nên kết quả luôn cần phải được kiểm tra lại. Nếu có sự khác biệt lớn giữa thực tế và model, thì hiệu suất của model trong thế giới lớn không được đảm bảo. Và ngay khi 2 thế giới có trùng nhau, thì một mẫu bất kỳ của data vẫn có thể bị cho kết quả sai. Cho nên hẫy nhớ ít nhất hai nguyên tắc sau đây.

1. Tính bất định của model không đảm bảo rằng là model tốt. Khi số lượng data tăng lên, model tung quả cầu sẽ khẳng định hơn tỉ lệ nước trên bề mặt. Nghĩa là đường cong sẽ cao lên và thu hẹp, giới hạn những khả năng của tỉ lệ trong khoảng hẹp hơn. Nhưng model -Bayesian hoặc non-Bayesian- có thể có độ tin cậy cao với một ước lượng, ngay cả khi model rất sai. Bởi vì ước lượng đặt điều kiện lên model. Những gì model nói với bạn, dưới một điều kiện cụ thể của model, nó rất đảm bảo rằng giá trị ước lượng ở khoảng hẹp. Với model khác, mọi thứ có thể không giống hoàn toàn.

2. Việc hướng dẫn và đánh giá model rất quan trọng. Hãy nghĩ lại rằng việc update model Bayesian có thể dùng với bất kỳ thứ tự nào của data mới. Ta có thể thay đổi thứ tự của các quan sát, chỉ cần tồn tại 6 W và 3 L, thì cuối cùng vẫn là cùng một đường cong các khả năng của tỉ lệ nước. Điều này đúng bởi vì model giả định thứ tự không liên quan vào suy luận. Khi mà một thứ gì đó không liên quan với cỗ máy, nó không ảnh hưởng trực tiếp đến suy luận. Nhưng nó có thể ảnh hưởng gián tiếp, nếu data thay đổi dựa trên thứ tự. Cho nên việc kiểm tra suy luận của model rất quan trọng khi có data mới. Việc này là một công việc tư duy, cần đến nhà phân tích và cộng đồng khoa học. Golem không làm được chuyện đó.

Chú ý rằng mục đích của lượng giá không phải sự thật đúng sai của giả định model. Chúng ta đều biết giả định model không bao giờ đúng, vì ta không biết quy trình tạo data thực tế là như thế nào. Cho nên không cần kiểm tra model có đúng hay không. Không chứng minh sai được model, không phải là model hoạt động tốt, mà chỉ là thất bại trong trí tưởng tượng của chúng ta. Hơn nữa, model không cần phải chính xác để tạo suy luận có tính chính xác cao và có giá trị. Mọi trường hợp của giả định model về phân phối của sai số hoặc tương tự, có thể bị xâm phạm trong thế giới lớn, nhưng model vẫn cho ra một ước lượng tốt và hữu ích. Bởi vì model thực chất là cỗ máy xử lý thông tin, cho nên một số khía cạnh ngạc nhiên của thông tin không được nắm bát bởi chuyển đổi vấn đề dưới hình thức của giả định.

Thay vào đó, nhiệm vụ của ta là kiểm tra tính khả thi của model cho vài ý định thôi. Có nghĩa là ta hỏi và trả lời các câu hỏi, ngoài tầm của model sau khi tạo. Cả câu hỏi và trả lời đều trong cùng một ngữ cảnh. Cho nên rất khó để cung cấp lời khuyên cụ thể. 

>**Nghĩ lại: thống kê làm giảm lạm phát.** Có lẽ Suy luận Bayesian là phương pháp suy luận phổ thông tốt nhất. Nhưng thực ra nó không mạnh như chúng ta nghĩ. Không có phương pháp tiếp cận nào cho két quả nhất quán. Không một phân nhánh khoa học thống kê toán học nào có thể tiếp cận trực tiếp với thế giới thực, bởi nó không phải proton. Nó không phải được phát hiện, mà là được chế tạo ra, như cây xẻng.

## <center>2.3 Các thành phần của model</center><a name="2.3"></a>

Giờ bạn đã xem cách model Bayesian chạy, hãy mở cỗ máy ra và xem kết cấu của nó. Xem xét 3 con số mà ta đếm trong phần trước.

1. Số các cách mỗi phỏng đoán có thể tạo ra quan sát
2. Số tích luỹ các cách mỗi phỏng đoán tạo được toàn bộ data
3. Khả năng của mỗi phỏng đoán

Mỗi con số trên đều có liên quan đến thuyết xác suất. Và cách làm thông thường mà ta dựng model thống kê gồm chọn các phân phối và các thiết bị cho mỗi phân phối đại diện cho con số tương đối cách con số được tạo ra.

### 2.3.1 Các biến số (variables)

Variables (var) là những ký tự đại số cho những giá trị khác nhau. Trong ngữ cảnh khoa học, var gồm những thứ mà ta cần suy luận, nhưng tỉ lệ hay tần suất, cũng như là thứ mà ta quan sát, data. Trong mô hình tung quả cầu, có 3 var.

Var đầu tiên là đối tượng ta cần suy luận, *p*, tỉ lệ nước bề mặt quả cầu. Var này ta không quan sát được. Kiểu var này gọi là **PARAMETER**. Mặc dù *p* không quan sát được, nhưng ta suy luận nó qua var khác.

Var khác là var quan sát được, là số đếm nước và đất. Gọi số đếm nước W và đất L. N là tổng số lần tung, N = W + L.

### 2.3.2 Định nghĩa

Khi ta liệt kê được các var, thì ta bắt đầu định nghĩa nó. Ta dựng model liên quan giữa các var. Hãy nhớ rằng, mục tiêu cuối cùng là đếm tất cả các cách mà data có thể xuất hiện, dưới giả định ban đầu. Có nghĩa là, trong model tung quả cầu, với mỗi giá trị khả thi của var không quan sát được, như *p*, ta tìm số cách tương đối - xác suất - mà giá trị của var quan sát được xảy ra. Và với mỗi var không quan sát được, ta định nghĩa xác suất tiền nghiệm (prior) với từng giá trị có thể của nó. 

#### 2.3.2.1 Var quan sát được.

Với số đếm nước W và đất L, ta định nghĩa với mỗi xác suất của *p*, ta tìm khả năng của nó với mọi kết hợp của W và L. Nó giống như ta đã đếm số viên bi. Với một giá trị cụ thể của *p* tương ứng với một khả năng nhất định của data.

Chúng ta không cần phải đếm tay, bằng cách dùng công thức toán học để cho ra kết quả đúng. Trong thống kê thông thường, hàm phân phối thường được gán vào var quan sát được gọi là **LIKELIHOOD**. Thuật ngữ này có ý nghĩa đặc biệt trong thống kê non-Bayesian. Chúng ta có thể làm những chuyện với phân phối mà thống kê non-Bayesian không cho phép. Cho nên tôi sẽ tránh dùng chữ *likelihood* mà chỉ nói đến phân phối của var. Nhưng khi một người nào nói *likelihood*, nó có nghĩa là hàm phân phối được gán với var quan sát được.

Trong ví dụ tung quả cầu, hàm mà chúng ta cần có thể suy trực tiếp từ câu chuyện data. Bắt đầu bằng liệt kê mọi sự kiện. Có 2 sự kiện: nước (W) và đất (L). Không có sự kiện khác, như quả cầu dính lên trần nhà. Ta quan sát một mẫu với cỡ mẫu là N (trong ví dụ là 9), ta phải trả lời được cách mà data xuất hiện, trong vũ trụ các mẫu có thể xảy ra với cùng cỡ mẫu. Có vẻ nó khó, nhưng khi bạn quen dần thì có thể làm điều này nhanh chóng.

Giả định của chúng ta thêm: (1) các lần tung đều độc lập với nhau (2) xác suất ra W đều như nhau với các lần tung. Thuyết xác suất cho ta một câu trả lời độc nhất, *phân phối binomial*. 

$$ Pr(W, L|p) = \frac{(W + L)!}{W!L!} p^W (1-p)^L $$

Hay còn đọc là:

>Số đếm nước W và đất L được phân phối binomial, với xác suất ra nước là p với mọi lần tung.

Có thể dùng code numpyro như sau:

```python
np.exp(dist.Binomial(total_count=9, probs=0.5).log_prob(6))
#  DeviceArray(0.16406256, dtype=float32)
```

Con số trên là số cách tương đối để được 6 nước và 3 đất, với *p*=0.5. Vậy nó làm công chuyện đếm số cách tương đối trong khu vườn. Thay đổi giá trị *p* để xem nó thay đổi như thế nào.

Nhưng chương sau, bạn sẽ thấy phân phối binomial rất đặc biệt, vì nó đại diện cho **MAXIMUM ENTROPY**,  để đếm các sự kiện nhị phân. "Maximum Entropy" nghe có vẻ không hay. Nó đại diện cho sự hỗn loạn. Thực ra nó nghĩa là phân phối không chứ thông tin khác ngoài: có 2 sự kiện, và xác suất của sự kiện là *p* và 1 - *p*. 

>**Nghĩ lại:** Vai trò của likelihood. Bài viết khá dài dòng về sự khác nhau giữa Bayesian và non-Bayesian. Nhìn ra sự khác nhau là tốt nhưng đôi lúc làm ta không chú ý với sự giống nhau về cơ bản. Cụ thể, chúng giống nhau ở giả định về phân phối được gán vào data, hàm likelihood. Suy luận từ likelihood lấy từ tất cả data, và khi cỡ mẫu tăng, nó càng quan trọng hơn. Đó giải thích tại sao nhiều quy trình ở Bayesian và non-Bayesian lại giống nhau.

#### 2.3.2.2 Var không quan sát được

Phân phối gán cho var quan sát được thì có var riêng của nó, ví dụ như số đếm W và L. Còn *p* là var không quan sát được, ta gọi nó là **PARAMETER**. Mặc dù không quan sát được, ta vẫn phải định nghĩa nó.

Trong bài sau sẽ có nhiều parameter hơn trong model. Trong thực hành, có nhiều câu hỏi thường gặp mà ta cần tìm hiểu trong data được trả lời trực tiếp bằng parameters:

- Trung bình của sự khác nhau giữa các nhóm điều trị?
- Mức độ liên quan của điều trị và kết quả?
- Hiệu ứng của điều trị trên một hiệp biến (covariate)?
- Phương sai khác nhau giữa các nhóm?

Bạn sẽ thấy rằng các câu hỏi này trở thành parameters trong phân phối mà ta gán cho data.

Với mọi parameter mà bạn cho vào cỗ máy Bayesian, bạn phải cho một phân phối tiền nghiệm cho nó, **PRIOR**. Một cỗ máy Bayesian cần phải có khả năng ban đầu gán cho từng giá trị có thể của parameter. Khi mà bạn có prior cho cỗ máy, giống như hình ở trên, cỗ máy sẽ học từng data một. Kết quả, mỗi ước lượng sẽ là prior cho bước tiếp theo. Nhưng nó không giải quyết được vấn đề tạo prior, bởi vì khi n=0, thời điểm ban đầu, cỗ máy vẫn cần có trạng thái sơ khởi với *p*: một đường thẳng ngang mô tả xác suất tương đồng với mọi khả năng của *p*.

Vậy prior đến từ đâu? Prior là những giả định về mặt kỹ thuật, được chọn để giúp cỗ máy học, cũng là giả định về mặt khoa học, phản ánh những gì ta biết về hiện tượng sự vật. Prior đường thẳng ngang khá là phổ biến, nhưng không phải prior tốt nhất.

Có một trường phái suy luận Bayesian nhấn mạnh cách chọn prior dựa trên niềm tin của người thống kê. Hướng suy luận Bayesian chủ quan này có trong vài kỹ thuật thống kê, triết học, kinh tế học, nhưng nó hiếm trong khoa học. Trong phân tích data khoa học tự nhiên và xã hội, prior được xem là một phần của model. Và nó được chọn, lượng giá, đổi mới như các thành phần khác của model. Thực tế, hai trường phái này phân tích dữ liệu gần như giống nhau.

Tuy nhiên bạn không nên hiểu rằng là phân tích thống kê không bắt buộc phải chủ quan, bởi vì vốn dĩ nó vậy, rất nhiều khoa học luôn kèm theo những quyết định mang tính chủ quan. Prior và phân tích Bayesian không chủ quan không kém so với likelihood và giả định lấy mẫu lặp lại trong các phép kiểm định. Các bạn nào có nhờ đến các chuyên gia thống kê đều thường không đồng ý phân tích bằng một phương pháp chung, dù đó là vấn đề đơn giản nhất. Thực vậy, suy luận thống kê dùng toán học không nghĩa là chỉ có một phương pháp duy nhất để thực hiện. Kỹ sư cũng dùng toán, bởi có nhiều cách để xây một cây cầu.

Sau tất cả ở trên, ta không có luật lệ nào chỉ được dùng 1 prior. Nếu bạn không có một lập luận chắc về prior nào đó, hãy thứ cái khác. Bởi vì prior là một giả định, nó cần được cân nhắc như những giả định khác, bằng cách thay đổi và kiểm tra độ nhạy của nó đối với suy luận. Không ai cần phải lập lời thề để dùng giả định, và ta cũng không phải nô lê của một giả định nào.

---
**Nghĩ nhiều hơn: Prior như phân phối.** Bạn có thể prior trong ví dụ trên như sau:

$$ Pr(p) = \frac{1}{1-0} = 1 $$

Prior là phân phối cho parameter. Nhìn chung, với một prior Uniform từ *a* đến *b*, xác suất tại mọi điểm trong khoảng là 1/(*b* - *a*). Nếu bạn thắc mắc tại sao mọi điểm trong khoảng đều có *p*=1, hãy nhớ rằng tất cả phân phối xác suất đều có tích phân là 1. Công thức 1/(*b* - *a*) đảm bảo diện tích dưới đường thẳng từ a đến b là 1.

---

>**Nghĩ lại: Data hay parameter?** Ta thường nghĩ data và parameter là 2 thực thể khác nhau. Data đo lường được; parameters thì không đo được và cần được ước lượng từ data. Khung quy trình Bayesian không phân biệt rõ 2 thực thể này. Đôi khi ta quan sát được variable, đôi khi không. Trong trường hợp đó, ta áp dụng chung một hàm phân phối. Kết quả là, cùng một giả định nhưng nó có thể là prior hoặc likelihood, dựa vào ngữ cảnh, mà không cần thay đổi model. Bạn sẽ thấy cách đo đạc sai số và mất data trong model ở các chương cuối.



>**Nghĩ lại: Khủng hoảng Prior.** Trong quá khứ, vài đối thủ của Suy luận Bayesian phản đối việc dùng prior ngãu nhiên. Prior thì đúng là linh hoạt, có thể mã hoá các trạng thái khác nhau của thông tin. Nếu prior có thể là bất kỳ thứ gì, thì bạn có thể điều khiển được kết quả? Thực tế là nó đúng, nhưng sau hàng thế kỷ trong lịch sử Bayesian, không ai dùng nó để lừa đảo cả. Nếu như bạn muốn lừa đảo bằng thống kê, bạn là một thằng ngốc, bởi nó rất dễ bị lật tẩy. Tốt hơn là dùng hệ thống likelihood kín đáo hơn. Hoặc tốt hơn là - đừng nghe theo lời khuyên này! - massage data, bỏ vài "outliers", hoặc là transform data.

> Sử dụng likelihood thì thường gặp hơn prior, nhưng likelihood thường yếu hơn, khó phát hiện các tương quan trong data. Ở khía cạnh này, cả Bayesian và non-Bayesian đều bị ảnh hưởng như nhau, bởi vì cả hai đều dựa nhiều vào hàm likelihood và dạng model thường gặp. Thực vậy, quy trình non-Bayesian không giả định prior, bởi vì quy trình cần phải quyết định thứ khác mà Bayesian không có, như cách phạt likelihood. Cách phạt này có thể xem như hàm loss của Bayesian.

### 2.3.3 Tạo model

Giờ ta tóm tắt model trong ví dụ như sau. Var quan sát được là W, L nằm trong phân phối Binomial, với N = W + L

$$ W \sim \text{Binomial}(N,p) $$

Var không quan sát được là *p* tương tự:

$$ p \sim \text{Uniform}(0,1) $$

Có nghĩa là *p* có một prior phẳng, với giá trị của *p* đều có khả năng như nhau, trong khoảng 0-1. Như ta biết, đây chưa phải prior tốt nhất, vì ta biết bề mặt Trái Đất có nhiều nước hơn đất liền, chỉ chưa biết tỉ lệ thôi.

## <center>2.4 Cho model hoạt động</center><a name="2.4"></a>

Sau khi đặt tên cho tất cả các variable và định nghĩa của chúng, model Bayesian sẽ update tất cả các prior thành **POSTERIOR**, hệ quả logic chính chủ. Với mỗi cặp độc nhất của data, likelihood, parameters, prior, sẽ cho ra một phân phối posterior độc nhất. Phân phối posterior gồm khả năng tương đối của từng giá trị của parameters, với điều kiện là data và model. Phân phối posterior có dạng phân phối prior của parameter, với điều kiện là data. Trong ví dụ, phân phối posterior là $Pr(p\|W, L)$, phân phối của mọi giá trị *p*, với điều kiện là W và L cụ thể mà ta đã quan sát được.

### 2.4.1 Bayes' theorem.

Định nghĩa toán học của phân phối posterior đến từ Bayes' theorem, là nguồn gốc của phân tích data Bayesian. Nhưng cái tên này thực ra là một mặt của thuyết xác suất. Như trong ví dụ, nó chỉ là tái hiện của phép nhân trong data, ở đó nó dùng phân phối để update. Nhưng nó vẫn là phép đếm.

Xác suất kết hợp (Joint probability) của data W và L với prior có parameter là *p*:

$$ Pr(W, L, p) = Pr(W,L|p)Pr(p) $$

Công thức nói rằng xác suất vừa có W, L, *p* là tích của xác suất W,L có điều kiện là *p*, và prior là xác suất Pr(*p*). Nó giống như là nói xác suất trời mưa và trời lạnh cùng một ngày bằng với xác suất trời mưa khi trời lạnh, nhân với xác suất trời lạnh. Nó chỉ là định nghĩa. Nó cũng được viết:

$$ Pr(W,L,p) = Pr(p|W,L) Pr(W,L) $$

Công thức trên chỉ đảo vị trí của điều kiện. Định nghĩa trên vẫn đúng. Nó giống như là nói xác suất trời mưa và trời lạnh cùng một ngày bằng với xác suất trời lạnh khi trời mưa, nhân với xác suất trời mưa.

Vậy:

$$  Pr(p|W,L) Pr(W,L) = Pr(W,L|p)Pr(p) $$

Ta muốn Pr(p \| W,L):

$$  Pr(p|W,L)  = \frac{Pr(W,L|p)Pr(p)}{Pr(W,L)} $$

Đây là Bayés' theorem, nó nói rằng xác suất của một *p* cụ thể, sau khi có data, thì bằng với tích của khả năng tương đối của data, với điều kiện là *p*, và xác suất ban đầu của *p*, chia cho Pr(W, L), mà tôi gọi là xác suất trung bình của data.

$$ \text{Posterior} = \frac{\text{Xác suất của data}\times \text{Prior}}{\text{Xác suất trung bình của data}} $$

Xác suất trung bình của data, Pr(W, L) có thể hơi rối. Nó thường được gọi là "bằng chứng" hay "Likelihood trung bình", nhưng chúng cũng không rõ ràng. Nó chính xác là xác suất trung bình của data, dựa trên prior. Nhiệm vụ chính của nó là chuẩn hoá posterior, để đảm bảo tổng (tích phân) là 1. 

$$ Pr(W, L) = E(Pr(W, L|p)) = \int Pr(W, L|p) Pr(p) dp $$

Kí hiệu E là *expection* - mong đợi. Trung bình thường được gọi là *marginal* trong toán học thống kê, nên bạn sẽ thấy phân phối này là *marginal likelihood*. Và Tích phân trên là công thức cụ thể để tính trung bình của một phân phối liên tục, ví dụ như có vô số khả năng các giá trị *p*.

Bài học quan trọng ở đây là posterior thì đồng dạng với tích của prior và xác suất của data. Tại sao? Vì với mỗi giá trị cụ thể *p*, số đếm các cách đi qua khu vườn phân nhánh là tích của số các nhánh prior và số đường đi. Phép nhân đơn giản hoá việc đếm. Xác suất trung bình ở mẫu số dùng để chuẩn hoá số đếm để chúng có tổng là 1. 

![figure 2-6](/assets/images/figure 2-6.png)

Hình trên mô tả tương tác giữa prior và xác suất của data. Prior ở bên trái nhân với xác suất của data ở bên giữa tạo ra posterior ở bên phải. Xác suất của data thì như nhau, prior thì khác nhau, và posterior cũng thay đổi theo.

>**Nghĩ lại: Phân tích Bayesian không phải chỉ Bayes' theorem.** Một đặc điểm của phân tích Bayesian, và suy luận Bayesian nói chung, là ở chỗ nó dùng Bayes' theorem. Đây là một nhận định sai. Suy luận bằng lý thuyết xác suất luôn phải dùng Bayes' theorem. Ví dụ hướng dẫn thường lấy test HIV, DNA không phải là thuần Bayesian. Vì các thành phần của phép tính là tần số của mẫu quan sát, phân tích non-Bayesian vẫn có thể làm được tương tự. Ngược lại, cách tiếp cận Bayesian dùng Bayes' theorem một cách tổng quát, để đánh giá tính bất định về thực thể lý thuyết không quan sát được, như parameter và model. Suy luận mạnh mẽ có thể thực hiện bởi quan điểm Bayesian và non-Bayesian, nhưng phải chấp nhận bị chỉnh sửa hoặc hi sinh.

### 2.4.2 Motor

Hãy nhớ lại model Bayesian là một cỗ máy, một con golem có thể tuỳ chỉnh được. Nó có sẵn các định nghĩa về likelihood, parameters, và prior. Và trong trái tim của nó là một motor xử lý data tạo ra posterior. Công việc này của motor có thể nghĩ là đặt điều kiện của prior vào data. Như đã giải thích phần trước, việc áp dụng điều kiện này là dựa trên lý thuyết xác suất, và nó tạo ra một posterior độc nhất dựa trên cặp giả định và quan sát.

Tuy nhiên, biết được cách làm toán học thường không giúp ích gì nhiều, bởi vì có nhiều model hay trong khoa học hiện nay không thể tính toán thông thường, không cần biết bạn giỏi toán đến đâu. Trong một ít model dùng nhiều hiện nay như linear regression có thể tính được bằng cách thông thường, bởi vì bạn đã ràng buộc prior ở dạng đặc biệt để có dùng áp vào công thức toán học. Chúng tôi cố gắng né tránh cách tạo model kiểu vậy, thay vào đó là động cơ có thể dùng được với mọi prior nào mà tốt nhất cho suy luận.

Động cơ này là những kỹ thuật ước lượng toán học tạo posterior dựa trên Bayes' theorem. Sách này nói về 3 kỹ thuật:

1. Grid approximation (ước lượng theo lưới)
2. Quadratic approximation (ước lượng đỉnh bằng phương trình bậc 2)
3. Markov chain Monte Carlo (MCMC)

Có nhiều động cơ khác và mới hơn, nhưng 3 kỹ thuật này rất thông dụng và hiệu quả. Hiểu 3 kỹ thuật này sẽ giúp bạn hiểu những kỹ thuật khác.

>**Nghĩ lại: bạn fit model như thế nào cũng là một phần của model.** Lúc đầu, tôi mô tả model là tập hợp prior và likelihood. Định nghĩa này là kinh điển. Nhưng trong thực hành, ta nên cân nhắc cách model fit data như một phần của model. Với bài toán đơn giản, như ví dụ tung quả cầu, các phép tính tìm mật độ xác suất của posterior là dễ dàng và không bao giờ sai. Trong bài toán phức tạp hơn, chi tiết cách fit data vào model làm ta nhận ra là các kỹ thuật tính toán cũng ảnh hưởng đến suy luận kết quả. Bởi vì có nhiều lỗi sai và thoả thuận khác nhau trong các kỹ thuật khác nhau. Cùng một model fit cùng một data nhưng dùng kỹ thuật khác sẽ tạo ra kết quả khác. Khi có gì đó trục trặc, mọi thành phần của cỗ máy đều cần phải được kiểm tra lại. Và golem dùng động cơ đó để hoạt động, và cũng là nô lệ cho các kỹ thuật chế tạo nó, tương tự như các priors và likelihoods mà chúng ta đã lập trình cho chúng.

### 2.4.3 Grid approximation

Là kỹ thuật đơn giản nhất để tính posterior. Mặc dù parameters thường là biến liên tục, có thể có vô số các giá trị, ta vẫn có thể ước lượng phân phối của nó bằng các lấy một số lượng mẫu có hạn. Với một giá trị *p'* cụ thể, thì rất dễ để tính xác suất posterior: chỉ cần nhân prior *p'* với likelihood tại *p'*. Lặp lại quy trình này với tất cả các giá trị *p* trong grid (lưới), bạn sẽ tạo ra được bức tranh tương đối của phân phối posterior.

Grid approx chủ yếu dùng để dùng để giảng dạy, để người học hiểu được bản chất của cập nhật kiểu Bayesian. Trong thế giới lớn, grid approx là không khả dụng. Vì khả năng mở rộng của nó rất kém, khi số lượng parameter tăng cao. 

Ta sẽ grid approx phân phối posterior trong ví dụ tung quả cầu:
1. Định nghĩa grid: chọn các điểm trong prior, là tập hợp có giới hạn của các giá trị *p* trong khoảng 0-1.
2. Tính xác suất prior của từng điểm trong grid.
3. Tính likelihood của từng điểm trong grid.
4. Tính xác suất posterior chưa chuẩn hoá của từng diểm trong grid, bằng cách nhân xác suất prior và likelihood.
5. Tính xác suất chuẩn hoá của posterior, bằng cách chia từng xác suất cho tổng của chúng.

```python
# define grid
p_grid = np.linspace(start=0, stop=1, num=20)

# define prior
prior = np.repeat(1, 20)

# compute likelihood at each value in grid
likelihood = np.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6))

# compute product of likelihood and prior
unstd_posterior = likelihood * prior

# standardize the posterior, so it sums to 1
posterior = unstd_posterior / np.sum(unstd_posterior)
```

Vẽ plot có 20 điểm trong grid

```python
plt.plot(p_grid, posterior, "-o")
plt.xlabel("probability of water")
plt.ylabel("posterior probability")
plt.title("20 points");
```

<svg height="329.052062pt" version="1.1" viewBox="0 0 469.349062 329.052062" width="469.349062pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style></defs><g id="figure_1"><g id="patch_1"><path d="M 0 329.052062 
L 469.349063 329.052062 
L 469.349063 0 
L 0 0 
z
" style="fill:#ffffff;"></path></g><g id="axes_1"><g id="patch_2"><path d="M 60.389063 286.2855 
L 462.149063 286.2855 
L 462.149063 25.3575 
L 60.389063 25.3575 
z
" style="fill:#eeeeee;"></path></g><g id="matplotlib.axis_1"><g id="xtick_1"><g id="line2d_1"><path clip-path="url(#p0e47278793)" d="M 78.650881 286.2855 
L 78.650881 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_2"></g><g id="text_1"><!-- 0.0 --><defs><path d="M 31.78125 66.40625 
Q 24.171875 66.40625 20.328125 58.90625 
Q 16.5 51.421875 16.5 36.375 
Q 16.5 21.390625 20.328125 13.890625 
Q 24.171875 6.390625 31.78125 6.390625 
Q 39.453125 6.390625 43.28125 13.890625 
Q 47.125 21.390625 47.125 36.375 
Q 47.125 51.421875 43.28125 58.90625 
Q 39.453125 66.40625 31.78125 66.40625 
z
M 31.78125 74.21875 
Q 44.046875 74.21875 50.515625 64.515625 
Q 56.984375 54.828125 56.984375 36.375 
Q 56.984375 17.96875 50.515625 8.265625 
Q 44.046875 -1.421875 31.78125 -1.421875 
Q 19.53125 -1.421875 13.0625 8.265625 
Q 6.59375 17.96875 6.59375 36.375 
Q 6.59375 54.828125 13.0625 64.515625 
Q 19.53125 74.21875 31.78125 74.21875 
z
" id="DejaVuSans-48"></path><path d="M 10.6875 12.40625 
L 21 12.40625 
L 21 0 
L 10.6875 0 
z
" id="DejaVuSans-46"></path></defs><g style="fill:#262626;" transform="translate(67.518693 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_2"><g id="line2d_3"><path clip-path="url(#p0e47278793)" d="M 151.698153 286.2855 
L 151.698153 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_4"></g><g id="text_2"><!-- 0.2 --><defs><path d="M 19.1875 8.296875 
L 53.609375 8.296875 
L 53.609375 0 
L 7.328125 0 
L 7.328125 8.296875 
Q 12.9375 14.109375 22.625 23.890625 
Q 32.328125 33.6875 34.8125 36.53125 
Q 39.546875 41.84375 41.421875 45.53125 
Q 43.3125 49.21875 43.3125 52.78125 
Q 43.3125 58.59375 39.234375 62.25 
Q 35.15625 65.921875 28.609375 65.921875 
Q 23.96875 65.921875 18.8125 64.3125 
Q 13.671875 62.703125 7.8125 59.421875 
L 7.8125 69.390625 
Q 13.765625 71.78125 18.9375 73 
Q 24.125 74.21875 28.421875 74.21875 
Q 39.75 74.21875 46.484375 68.546875 
Q 53.21875 62.890625 53.21875 53.421875 
Q 53.21875 48.921875 51.53125 44.890625 
Q 49.859375 40.875 45.40625 35.40625 
Q 44.1875 33.984375 37.640625 27.21875 
Q 31.109375 20.453125 19.1875 8.296875 
z
" id="DejaVuSans-50"></path></defs><g style="fill:#262626;" transform="translate(140.565966 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="xtick_3"><g id="line2d_5"><path clip-path="url(#p0e47278793)" d="M 224.745426 286.2855 
L 224.745426 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_6"></g><g id="text_3"><!-- 0.4 --><defs><path d="M 37.796875 64.3125 
L 12.890625 25.390625 
L 37.796875 25.390625 
z
M 35.203125 72.90625 
L 47.609375 72.90625 
L 47.609375 25.390625 
L 58.015625 25.390625 
L 58.015625 17.1875 
L 47.609375 17.1875 
L 47.609375 0 
L 37.796875 0 
L 37.796875 17.1875 
L 4.890625 17.1875 
L 4.890625 26.703125 
z
" id="DejaVuSans-52"></path></defs><g style="fill:#262626;" transform="translate(213.613239 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="xtick_4"><g id="line2d_7"><path clip-path="url(#p0e47278793)" d="M 297.792699 286.2855 
L 297.792699 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_8"></g><g id="text_4"><!-- 0.6 --><defs><path d="M 33.015625 40.375 
Q 26.375 40.375 22.484375 35.828125 
Q 18.609375 31.296875 18.609375 23.390625 
Q 18.609375 15.53125 22.484375 10.953125 
Q 26.375 6.390625 33.015625 6.390625 
Q 39.65625 6.390625 43.53125 10.953125 
Q 47.40625 15.53125 47.40625 23.390625 
Q 47.40625 31.296875 43.53125 35.828125 
Q 39.65625 40.375 33.015625 40.375 
z
M 52.59375 71.296875 
L 52.59375 62.3125 
Q 48.875 64.0625 45.09375 64.984375 
Q 41.3125 65.921875 37.59375 65.921875 
Q 27.828125 65.921875 22.671875 59.328125 
Q 17.53125 52.734375 16.796875 39.40625 
Q 19.671875 43.65625 24.015625 45.921875 
Q 28.375 48.1875 33.59375 48.1875 
Q 44.578125 48.1875 50.953125 41.515625 
Q 57.328125 34.859375 57.328125 23.390625 
Q 57.328125 12.15625 50.6875 5.359375 
Q 44.046875 -1.421875 33.015625 -1.421875 
Q 20.359375 -1.421875 13.671875 8.265625 
Q 6.984375 17.96875 6.984375 36.375 
Q 6.984375 53.65625 15.1875 63.9375 
Q 23.390625 74.21875 37.203125 74.21875 
Q 40.921875 74.21875 44.703125 73.484375 
Q 48.484375 72.75 52.59375 71.296875 
z
" id="DejaVuSans-54"></path></defs><g style="fill:#262626;" transform="translate(286.660511 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-54"></use></g></g></g><g id="xtick_5"><g id="line2d_9"><path clip-path="url(#p0e47278793)" d="M 370.839972 286.2855 
L 370.839972 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_10"></g><g id="text_5"><!-- 0.8 --><defs><path d="M 31.78125 34.625 
Q 24.75 34.625 20.71875 30.859375 
Q 16.703125 27.09375 16.703125 20.515625 
Q 16.703125 13.921875 20.71875 10.15625 
Q 24.75 6.390625 31.78125 6.390625 
Q 38.8125 6.390625 42.859375 10.171875 
Q 46.921875 13.96875 46.921875 20.515625 
Q 46.921875 27.09375 42.890625 30.859375 
Q 38.875 34.625 31.78125 34.625 
z
M 21.921875 38.8125 
Q 15.578125 40.375 12.03125 44.71875 
Q 8.5 49.078125 8.5 55.328125 
Q 8.5 64.0625 14.71875 69.140625 
Q 20.953125 74.21875 31.78125 74.21875 
Q 42.671875 74.21875 48.875 69.140625 
Q 55.078125 64.0625 55.078125 55.328125 
Q 55.078125 49.078125 51.53125 44.71875 
Q 48 40.375 41.703125 38.8125 
Q 48.828125 37.15625 52.796875 32.3125 
Q 56.78125 27.484375 56.78125 20.515625 
Q 56.78125 9.90625 50.3125 4.234375 
Q 43.84375 -1.421875 31.78125 -1.421875 
Q 19.734375 -1.421875 13.25 4.234375 
Q 6.78125 9.90625 6.78125 20.515625 
Q 6.78125 27.484375 10.78125 32.3125 
Q 14.796875 37.15625 21.921875 38.8125 
z
M 18.3125 54.390625 
Q 18.3125 48.734375 21.84375 45.5625 
Q 25.390625 42.390625 31.78125 42.390625 
Q 38.140625 42.390625 41.71875 45.5625 
Q 45.3125 48.734375 45.3125 54.390625 
Q 45.3125 60.0625 41.71875 63.234375 
Q 38.140625 66.40625 31.78125 66.40625 
Q 25.390625 66.40625 21.84375 63.234375 
Q 18.3125 60.0625 18.3125 54.390625 
z
" id="DejaVuSans-56"></path></defs><g style="fill:#262626;" transform="translate(359.707784 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-56"></use></g></g></g><g id="xtick_6"><g id="line2d_11"><path clip-path="url(#p0e47278793)" d="M 443.887244 286.2855 
L 443.887244 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_12"></g><g id="text_6"><!-- 1.0 --><defs><path d="M 12.40625 8.296875 
L 28.515625 8.296875 
L 28.515625 63.921875 
L 10.984375 60.40625 
L 10.984375 69.390625 
L 28.421875 72.90625 
L 38.28125 72.90625 
L 38.28125 8.296875 
L 54.390625 8.296875 
L 54.390625 0 
L 12.40625 0 
z
" id="DejaVuSans-49"></path></defs><g style="fill:#262626;" transform="translate(432.755057 300.423312)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="text_7"><!-- probability of water --><defs><path d="M 18.109375 8.203125 
L 18.109375 -20.796875 
L 9.078125 -20.796875 
L 9.078125 54.6875 
L 18.109375 54.6875 
L 18.109375 46.390625 
Q 20.953125 51.265625 25.265625 53.625 
Q 29.59375 56 35.59375 56 
Q 45.5625 56 51.78125 48.09375 
Q 58.015625 40.1875 58.015625 27.296875 
Q 58.015625 14.40625 51.78125 6.484375 
Q 45.5625 -1.421875 35.59375 -1.421875 
Q 29.59375 -1.421875 25.265625 0.953125 
Q 20.953125 3.328125 18.109375 8.203125 
z
M 48.6875 27.296875 
Q 48.6875 37.203125 44.609375 42.84375 
Q 40.53125 48.484375 33.40625 48.484375 
Q 26.265625 48.484375 22.1875 42.84375 
Q 18.109375 37.203125 18.109375 27.296875 
Q 18.109375 17.390625 22.1875 11.75 
Q 26.265625 6.109375 33.40625 6.109375 
Q 40.53125 6.109375 44.609375 11.75 
Q 48.6875 17.390625 48.6875 27.296875 
z
" id="DejaVuSans-112"></path><path d="M 41.109375 46.296875 
Q 39.59375 47.171875 37.8125 47.578125 
Q 36.03125 48 33.890625 48 
Q 26.265625 48 22.1875 43.046875 
Q 18.109375 38.09375 18.109375 28.8125 
L 18.109375 0 
L 9.078125 0 
L 9.078125 54.6875 
L 18.109375 54.6875 
L 18.109375 46.1875 
Q 20.953125 51.171875 25.484375 53.578125 
Q 30.03125 56 36.53125 56 
Q 37.453125 56 38.578125 55.875 
Q 39.703125 55.765625 41.0625 55.515625 
z
" id="DejaVuSans-114"></path><path d="M 30.609375 48.390625 
Q 23.390625 48.390625 19.1875 42.75 
Q 14.984375 37.109375 14.984375 27.296875 
Q 14.984375 17.484375 19.15625 11.84375 
Q 23.34375 6.203125 30.609375 6.203125 
Q 37.796875 6.203125 41.984375 11.859375 
Q 46.1875 17.53125 46.1875 27.296875 
Q 46.1875 37.015625 41.984375 42.703125 
Q 37.796875 48.390625 30.609375 48.390625 
z
M 30.609375 56 
Q 42.328125 56 49.015625 48.375 
Q 55.71875 40.765625 55.71875 27.296875 
Q 55.71875 13.875 49.015625 6.21875 
Q 42.328125 -1.421875 30.609375 -1.421875 
Q 18.84375 -1.421875 12.171875 6.21875 
Q 5.515625 13.875 5.515625 27.296875 
Q 5.515625 40.765625 12.171875 48.375 
Q 18.84375 56 30.609375 56 
z
" id="DejaVuSans-111"></path><path d="M 48.6875 27.296875 
Q 48.6875 37.203125 44.609375 42.84375 
Q 40.53125 48.484375 33.40625 48.484375 
Q 26.265625 48.484375 22.1875 42.84375 
Q 18.109375 37.203125 18.109375 27.296875 
Q 18.109375 17.390625 22.1875 11.75 
Q 26.265625 6.109375 33.40625 6.109375 
Q 40.53125 6.109375 44.609375 11.75 
Q 48.6875 17.390625 48.6875 27.296875 
z
M 18.109375 46.390625 
Q 20.953125 51.265625 25.265625 53.625 
Q 29.59375 56 35.59375 56 
Q 45.5625 56 51.78125 48.09375 
Q 58.015625 40.1875 58.015625 27.296875 
Q 58.015625 14.40625 51.78125 6.484375 
Q 45.5625 -1.421875 35.59375 -1.421875 
Q 29.59375 -1.421875 25.265625 0.953125 
Q 20.953125 3.328125 18.109375 8.203125 
L 18.109375 0 
L 9.078125 0 
L 9.078125 75.984375 
L 18.109375 75.984375 
z
" id="DejaVuSans-98"></path><path d="M 34.28125 27.484375 
Q 23.390625 27.484375 19.1875 25 
Q 14.984375 22.515625 14.984375 16.5 
Q 14.984375 11.71875 18.140625 8.90625 
Q 21.296875 6.109375 26.703125 6.109375 
Q 34.1875 6.109375 38.703125 11.40625 
Q 43.21875 16.703125 43.21875 25.484375 
L 43.21875 27.484375 
z
M 52.203125 31.203125 
L 52.203125 0 
L 43.21875 0 
L 43.21875 8.296875 
Q 40.140625 3.328125 35.546875 0.953125 
Q 30.953125 -1.421875 24.3125 -1.421875 
Q 15.921875 -1.421875 10.953125 3.296875 
Q 6 8.015625 6 15.921875 
Q 6 25.140625 12.171875 29.828125 
Q 18.359375 34.515625 30.609375 34.515625 
L 43.21875 34.515625 
L 43.21875 35.40625 
Q 43.21875 41.609375 39.140625 45 
Q 35.0625 48.390625 27.6875 48.390625 
Q 23 48.390625 18.546875 47.265625 
Q 14.109375 46.140625 10.015625 43.890625 
L 10.015625 52.203125 
Q 14.9375 54.109375 19.578125 55.046875 
Q 24.21875 56 28.609375 56 
Q 40.484375 56 46.34375 49.84375 
Q 52.203125 43.703125 52.203125 31.203125 
z
" id="DejaVuSans-97"></path><path d="M 9.421875 54.6875 
L 18.40625 54.6875 
L 18.40625 0 
L 9.421875 0 
z
M 9.421875 75.984375 
L 18.40625 75.984375 
L 18.40625 64.59375 
L 9.421875 64.59375 
z
" id="DejaVuSans-105"></path><path d="M 9.421875 75.984375 
L 18.40625 75.984375 
L 18.40625 0 
L 9.421875 0 
z
" id="DejaVuSans-108"></path><path d="M 18.3125 70.21875 
L 18.3125 54.6875 
L 36.8125 54.6875 
L 36.8125 47.703125 
L 18.3125 47.703125 
L 18.3125 18.015625 
Q 18.3125 11.328125 20.140625 9.421875 
Q 21.96875 7.515625 27.59375 7.515625 
L 36.8125 7.515625 
L 36.8125 0 
L 27.59375 0 
Q 17.1875 0 13.234375 3.875 
Q 9.28125 7.765625 9.28125 18.015625 
L 9.28125 47.703125 
L 2.6875 47.703125 
L 2.6875 54.6875 
L 9.28125 54.6875 
L 9.28125 70.21875 
z
" id="DejaVuSans-116"></path><path d="M 32.171875 -5.078125 
Q 28.375 -14.84375 24.75 -17.8125 
Q 21.140625 -20.796875 15.09375 -20.796875 
L 7.90625 -20.796875 
L 7.90625 -13.28125 
L 13.1875 -13.28125 
Q 16.890625 -13.28125 18.9375 -11.515625 
Q 21 -9.765625 23.484375 -3.21875 
L 25.09375 0.875 
L 2.984375 54.6875 
L 12.5 54.6875 
L 29.59375 11.921875 
L 46.6875 54.6875 
L 56.203125 54.6875 
z
" id="DejaVuSans-121"></path><path id="DejaVuSans-32"></path><path d="M 37.109375 75.984375 
L 37.109375 68.5 
L 28.515625 68.5 
Q 23.6875 68.5 21.796875 66.546875 
Q 19.921875 64.59375 19.921875 59.515625 
L 19.921875 54.6875 
L 34.71875 54.6875 
L 34.71875 47.703125 
L 19.921875 47.703125 
L 19.921875 0 
L 10.890625 0 
L 10.890625 47.703125 
L 2.296875 47.703125 
L 2.296875 54.6875 
L 10.890625 54.6875 
L 10.890625 58.5 
Q 10.890625 67.625 15.140625 71.796875 
Q 19.390625 75.984375 28.609375 75.984375 
z
" id="DejaVuSans-102"></path><path d="M 4.203125 54.6875 
L 13.1875 54.6875 
L 24.421875 12.015625 
L 35.59375 54.6875 
L 46.1875 54.6875 
L 57.421875 12.015625 
L 68.609375 54.6875 
L 77.59375 54.6875 
L 63.28125 0 
L 52.6875 0 
L 40.921875 44.828125 
L 29.109375 0 
L 18.5 0 
z
" id="DejaVuSans-119"></path><path d="M 56.203125 29.59375 
L 56.203125 25.203125 
L 14.890625 25.203125 
Q 15.484375 15.921875 20.484375 11.0625 
Q 25.484375 6.203125 34.421875 6.203125 
Q 39.59375 6.203125 44.453125 7.46875 
Q 49.3125 8.734375 54.109375 11.28125 
L 54.109375 2.78125 
Q 49.265625 0.734375 44.1875 -0.34375 
Q 39.109375 -1.421875 33.890625 -1.421875 
Q 20.796875 -1.421875 13.15625 6.1875 
Q 5.515625 13.8125 5.515625 26.8125 
Q 5.515625 40.234375 12.765625 48.109375 
Q 20.015625 56 32.328125 56 
Q 43.359375 56 49.78125 48.890625 
Q 56.203125 41.796875 56.203125 29.59375 
z
M 47.21875 32.234375 
Q 47.125 39.59375 43.09375 43.984375 
Q 39.0625 48.390625 32.421875 48.390625 
Q 24.90625 48.390625 20.390625 44.140625 
Q 15.875 39.890625 15.1875 32.171875 
z
" id="DejaVuSans-101"></path></defs><g style="fill:#262626;" transform="translate(187.724531 318.732531)scale(0.15 -0.15)"><use xlink:href="#DejaVuSans-112"></use><use x="63.476562" xlink:href="#DejaVuSans-114"></use><use x="104.558594" xlink:href="#DejaVuSans-111"></use><use x="165.740234" xlink:href="#DejaVuSans-98"></use><use x="229.216797" xlink:href="#DejaVuSans-97"></use><use x="290.496094" xlink:href="#DejaVuSans-98"></use><use x="353.972656" xlink:href="#DejaVuSans-105"></use><use x="381.755859" xlink:href="#DejaVuSans-108"></use><use x="409.539062" xlink:href="#DejaVuSans-105"></use><use x="437.322266" xlink:href="#DejaVuSans-116"></use><use x="476.53125" xlink:href="#DejaVuSans-121"></use><use x="535.710938" xlink:href="#DejaVuSans-32"></use><use x="567.498047" xlink:href="#DejaVuSans-111"></use><use x="628.679688" xlink:href="#DejaVuSans-102"></use><use x="663.884766" xlink:href="#DejaVuSans-32"></use><use x="695.671875" xlink:href="#DejaVuSans-119"></use><use x="777.458984" xlink:href="#DejaVuSans-97"></use><use x="838.738281" xlink:href="#DejaVuSans-116"></use><use x="877.947266" xlink:href="#DejaVuSans-101"></use><use x="939.470703" xlink:href="#DejaVuSans-114"></use></g></g></g><g id="matplotlib.axis_2"><g id="ytick_1"><g id="line2d_13"><path clip-path="url(#p0e47278793)" d="M 60.389063 274.425136 
L 462.149063 274.425136 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_14"></g><g id="text_8"><!-- 0.00 --><g style="fill:#262626;" transform="translate(25.717188 279.744043)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="ytick_2"><g id="line2d_15"><path clip-path="url(#p0e47278793)" d="M 60.389063 241.210944 
L 462.149063 241.210944 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_16"></g><g id="text_9"><!-- 0.02 --><g style="fill:#262626;" transform="translate(25.717188 246.52985)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="ytick_3"><g id="line2d_17"><path clip-path="url(#p0e47278793)" d="M 60.389063 207.996751 
L 462.149063 207.996751 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_18"></g><g id="text_10"><!-- 0.04 --><g style="fill:#262626;" transform="translate(25.717188 213.315657)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="ytick_4"><g id="line2d_19"><path clip-path="url(#p0e47278793)" d="M 60.389063 174.782558 
L 462.149063 174.782558 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_20"></g><g id="text_11"><!-- 0.06 --><g style="fill:#262626;" transform="translate(25.717188 180.101465)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-54"></use></g></g></g><g id="ytick_5"><g id="line2d_21"><path clip-path="url(#p0e47278793)" d="M 60.389063 141.568366 
L 462.149063 141.568366 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_22"></g><g id="text_12"><!-- 0.08 --><g style="fill:#262626;" transform="translate(25.717188 146.887272)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-56"></use></g></g></g><g id="ytick_6"><g id="line2d_23"><path clip-path="url(#p0e47278793)" d="M 60.389063 108.354173 
L 462.149063 108.354173 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_24"></g><g id="text_13"><!-- 0.10 --><g style="fill:#262626;" transform="translate(25.717188 113.673079)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-49"></use><use x="159.033203" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="ytick_7"><g id="line2d_25"><path clip-path="url(#p0e47278793)" d="M 60.389063 75.13998 
L 462.149063 75.13998 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_26"></g><g id="text_14"><!-- 0.12 --><g style="fill:#262626;" transform="translate(25.717188 80.458887)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-49"></use><use x="159.033203" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="ytick_8"><g id="line2d_27"><path clip-path="url(#p0e47278793)" d="M 60.389063 41.925788 
L 462.149063 41.925788 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_28"></g><g id="text_15"><!-- 0.14 --><g style="fill:#262626;" transform="translate(25.717188 47.244694)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-49"></use><use x="159.033203" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="text_16"><!-- posterior probability --><defs><path d="M 44.28125 53.078125 
L 44.28125 44.578125 
Q 40.484375 46.53125 36.375 47.5 
Q 32.28125 48.484375 27.875 48.484375 
Q 21.1875 48.484375 17.84375 46.4375 
Q 14.5 44.390625 14.5 40.28125 
Q 14.5 37.15625 16.890625 35.375 
Q 19.28125 33.59375 26.515625 31.984375 
L 29.59375 31.296875 
Q 39.15625 29.25 43.1875 25.515625 
Q 47.21875 21.78125 47.21875 15.09375 
Q 47.21875 7.46875 41.1875 3.015625 
Q 35.15625 -1.421875 24.609375 -1.421875 
Q 20.21875 -1.421875 15.453125 -0.5625 
Q 10.6875 0.296875 5.421875 2 
L 5.421875 11.28125 
Q 10.40625 8.6875 15.234375 7.390625 
Q 20.0625 6.109375 24.8125 6.109375 
Q 31.15625 6.109375 34.5625 8.28125 
Q 37.984375 10.453125 37.984375 14.40625 
Q 37.984375 18.0625 35.515625 20.015625 
Q 33.0625 21.96875 24.703125 23.78125 
L 21.578125 24.515625 
Q 13.234375 26.265625 9.515625 29.90625 
Q 5.8125 33.546875 5.8125 39.890625 
Q 5.8125 47.609375 11.28125 51.796875 
Q 16.75 56 26.8125 56 
Q 31.78125 56 36.171875 55.265625 
Q 40.578125 54.546875 44.28125 53.078125 
z
" id="DejaVuSans-115"></path></defs><g style="fill:#262626;" transform="translate(18.597656 232.036734)rotate(-90)scale(0.15 -0.15)"><use xlink:href="#DejaVuSans-112"></use><use x="63.476562" xlink:href="#DejaVuSans-111"></use><use x="124.658203" xlink:href="#DejaVuSans-115"></use><use x="176.757812" xlink:href="#DejaVuSans-116"></use><use x="215.966797" xlink:href="#DejaVuSans-101"></use><use x="277.490234" xlink:href="#DejaVuSans-114"></use><use x="318.603516" xlink:href="#DejaVuSans-105"></use><use x="346.386719" xlink:href="#DejaVuSans-111"></use><use x="407.568359" xlink:href="#DejaVuSans-114"></use><use x="448.681641" xlink:href="#DejaVuSans-32"></use><use x="480.46875" xlink:href="#DejaVuSans-112"></use><use x="543.945312" xlink:href="#DejaVuSans-114"></use><use x="585.027344" xlink:href="#DejaVuSans-111"></use><use x="646.208984" xlink:href="#DejaVuSans-98"></use><use x="709.685547" xlink:href="#DejaVuSans-97"></use><use x="770.964844" xlink:href="#DejaVuSans-98"></use><use x="834.441406" xlink:href="#DejaVuSans-105"></use><use x="862.224609" xlink:href="#DejaVuSans-108"></use><use x="890.007812" xlink:href="#DejaVuSans-105"></use><use x="917.791016" xlink:href="#DejaVuSans-116"></use><use x="957" xlink:href="#DejaVuSans-121"></use></g></g></g><g id="line2d_29"><path clip-path="url(#p0e47278793)" d="M 78.650881 274.425136 
L 97.873847 274.423809 
L 117.096814 274.353598 
L 136.319778 273.745774 
L 155.542747 271.279939 
L 174.765717 264.670332 
L 193.988675 251.1039 
L 213.211645 228.171474 
L 232.434614 195.041315 
L 251.657583 153.513183 
L 270.880553 108.565159 
L 290.103511 68.058167 
L 309.32647 41.403809 
L 328.54945 37.217864 
L 347.772408 60.287741 
L 366.995389 108.565233 
L 386.218347 171.363324 
L 405.441306 230.491443 
L 424.664286 266.686785 
L 443.887244 274.425136 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path><defs><path d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
" id="mf4cbdd6d80" style="stroke:#2a2eec;"></path></defs><g clip-path="url(#p0e47278793)"><use style="fill:#2a2eec;stroke:#2a2eec;" x="78.650881" xlink:href="#mf4cbdd6d80" y="274.425136"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="97.873847" xlink:href="#mf4cbdd6d80" y="274.423809"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="117.096814" xlink:href="#mf4cbdd6d80" y="274.353598"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="136.319778" xlink:href="#mf4cbdd6d80" y="273.745774"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="155.542747" xlink:href="#mf4cbdd6d80" y="271.279939"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="174.765717" xlink:href="#mf4cbdd6d80" y="264.670332"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="193.988675" xlink:href="#mf4cbdd6d80" y="251.1039"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="213.211645" xlink:href="#mf4cbdd6d80" y="228.171474"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="232.434614" xlink:href="#mf4cbdd6d80" y="195.041315"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="251.657583" xlink:href="#mf4cbdd6d80" y="153.513183"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="270.880553" xlink:href="#mf4cbdd6d80" y="108.565159"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="290.103511" xlink:href="#mf4cbdd6d80" y="68.058167"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="309.32647" xlink:href="#mf4cbdd6d80" y="41.403809"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="328.54945" xlink:href="#mf4cbdd6d80" y="37.217864"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="347.772408" xlink:href="#mf4cbdd6d80" y="60.287741"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="366.995389" xlink:href="#mf4cbdd6d80" y="108.565233"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="386.218347" xlink:href="#mf4cbdd6d80" y="171.363324"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="405.441306" xlink:href="#mf4cbdd6d80" y="230.491443"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="424.664286" xlink:href="#mf4cbdd6d80" y="266.686785"></use><use style="fill:#2a2eec;stroke:#2a2eec;" x="443.887244" xlink:href="#mf4cbdd6d80" y="274.425136"></use></g></g><g id="patch_3"><path d="M 60.389063 286.2855 
L 60.389063 25.3575 
" style="fill:none;"></path></g><g id="patch_4"><path d="M 462.149063 286.2855 
L 462.149063 25.3575 
" style="fill:none;"></path></g><g id="patch_5"><path d="M 60.389063 286.2855 
L 462.149063 286.2855 
" style="fill:none;"></path></g><g id="patch_6"><path d="M 60.389063 25.3575 
L 462.149063 25.3575 
" style="fill:none;"></path></g><g id="text_17"><!-- 20 points --><defs><path d="M 54.890625 33.015625 
L 54.890625 0 
L 45.90625 0 
L 45.90625 32.71875 
Q 45.90625 40.484375 42.875 44.328125 
Q 39.84375 48.1875 33.796875 48.1875 
Q 26.515625 48.1875 22.3125 43.546875 
Q 18.109375 38.921875 18.109375 30.90625 
L 18.109375 0 
L 9.078125 0 
L 9.078125 54.6875 
L 18.109375 54.6875 
L 18.109375 46.1875 
Q 21.34375 51.125 25.703125 53.5625 
Q 30.078125 56 35.796875 56 
Q 45.21875 56 50.046875 50.171875 
Q 54.890625 44.34375 54.890625 33.015625 
z
" id="DejaVuSans-110"></path></defs><g style="fill:#262626;" transform="translate(223.976563 19.3575)scale(0.16 -0.16)"><use xlink:href="#DejaVuSans-50"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-32"></use><use x="159.033203" xlink:href="#DejaVuSans-112"></use><use x="222.509766" xlink:href="#DejaVuSans-111"></use><use x="283.691406" xlink:href="#DejaVuSans-105"></use><use x="311.474609" xlink:href="#DejaVuSans-110"></use><use x="374.853516" xlink:href="#DejaVuSans-116"></use><use x="414.0625" xlink:href="#DejaVuSans-115"></use></g></g></g></g><defs><clipPath id="p0e47278793"><rect height="260.928" width="401.76" x="60.389063" y="25.3575"></rect></clipPath></defs></svg>

Số điểm càng nhiều thì chính xác càng cao. Bạn có thể thử 100000 điểm, nhưng suy luận sẽ không khác nhiều so với 100 điểm.

Hãy thử những prior khác nhau:

```python
prior = np.where(p_grid < 0.5, 0, 1)
prior = np.exp(-5 * abs(p_grid - 0.5))
```

### 2.4.4 Quadratic Approximation

Ta sẽ dùng grid approx cho các phần còn lại trong chương này, nhưng chắc chắn bạn phải dùng kỹ thuật ước lượng khác, với giả định khó khăn hơn. Lý do là với các giá trị độc nhất trong grid tăng rất nhanh khi số lượng parameter lớn. Với model tung quả cầu chỉ 1 parameter, không có khó khăn gì để tính grid 100 hay 1000 giá trị. Nhưng với 2 parameters, bạn có $100^2$ giá trị để thực hiện phép tính. Với 10 parameters, con số này thật khủng khiếp. Thời đại này, model với hàng trăm hay hàng ngàn parameters là thường gặp. Kỹ thuật grid approx này không phù hợp nữa.

Kỹ thuật **QUADRATIC APPROXIMATION** khá hữu ích. Trong điều kiện thông thường, vùng đỉnh của phân phối posterior gần như là dạng Gaussian - hay "phân phối bình thường". Điều này có nghĩa là phân phối posterior có thể ước lượng một phân phối Gaussian. Phân phối Gaussian rất tiện bởi nó chỉ được biểu diễn với 2 parameters: trung bình và phương sai.

Kỹ thuật này gọi là quadratic vì logarit của phân phối Gaussian là hình parabola, là phương trình bậc 2. Ước lượng này đại diện log-posterior bằng parabola.

Ta dùng quad approx trong rất nhiều bài. Ví dụ như Linear regression, kỹ thuật này rất hiệu quả. Thậm chí, nó còn có thể chính xác, chứ không phải ước lượng. Về phương diện tính toán, quad approx không tốn kém nhiều, ít hơn so với grid approx và MCMC. 

Kỹ thuật này gồm 2 bước:
1. Tìm đỉnh của posterior. Phần mềm thường dùng các thuật toán tối ưu hoá để "leo" lên đỉnh.
2. Ước lượng độ cong của các điểm quanh đỉnh. Độ cong này thường là đủ để tính quad approx cho toàn bộ phân phối posterior.

```python
def model(W, L):
    p = numpyro.sample("p", dist.Uniform(0, 1))  # uniform prior
    numpyro.sample("W", dist.Binomial(W + L, p), obs=W)  # binomial likelihood

guide = AutoLaplaceApproximation(model)
svi = SVI(model, guide, optim.Adam(1), AutoContinuousELBO(), W=6, L=3)
init_state = svi.init(PRNGKey(0))
state = lax.fori_loop(0, 1000, lambda i, x: svi.update(x)[0], init_state)
params = svi.get_params(state)

# display summary of quadratic approximation
samples = guide.sample_posterior(PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
```

|  |mean|std |median|5.5%|94.5%|n_eff |r_hat|
|--|----|----|------|----|-----|------|-----|
|p |0.62|0.14|0.63  |0.41|0.84 |845.27|1.00 |

Trung bình là 0.62, độ lệch chuẩn là 0.14, khoảng tin cậy 89% là 0.63-0.84. Ta biết posterior chính xác là Beta(7,4) (dùng conjugate prior) có trung bình là 0.6667. Vậy với một mẫu 6W, 3L thì quad approx chưa chính xác lắm. Nhưng khi số mẫu tăng lên, như nhân 2, nhân 4 số lần tung, thì nó sẽ cho ra đúng trung bình của posterior. 

![](/assets/images/figure 2-8.png)

Hiện tượng này rất thường gặp trong quad approx. Đó là lý do tại sao quy trình thống kê kinh điển thường rất sợ cỡ mẫu nhỏ: những quy trình dùng quad approx chỉ an toàn khi cỡ mẫu lớn. Nhưng sự cải thiện theo cỡ mẫu này cũng tuỳ vào nhiều yếu tố. Trong một số model, kỹ thuật quad approx vẫn sai nhiều với cỡ mẫu hàng ngàn.

Sử dựng Quad approx trong Bayesian vẫn có hạn chế tương tự. Nhưng bạn vẫn có nhiều thuật toán khác nếu bạn nghi ngờ nó. Nhắc lại, grid approx hoạt động rất tốt với cỡ mẫu nhỏ, bởi vì model lúc ấy đơn giản và ít các phép tính. Bạn cũng có thể MCMC, được giới thiệu tiếp theo.

>**Nghĩ lại: ước lượng maximum likelihood.** Quad approx, với prior là uniform, thường được tương đương với **MAXIMUM LIKELIHOOD ESTIMATE (MLE)** và standard error của nó. MLE rất thường gặp trong ước lượng parameter non-Bayesian. Sự tương ứng này vừa là một lời chúc vừa là một lời nguyền. Nó là lời chúc, vì nó cho phép chúng ta tái diễn giải nhiều model non-bayesian đã công bố. Nó là lời nguyền, bởi vì MLE có vài khuyết điểm, và quad approx cũng như thế. Ta sẽ tìme hiểu thêm, và nó là một trong những nguyên nhân ta dùng Markov chain Monte Carlo.

---

**Nghĩ nhiều hơn: Hessian đang tới.** Đôi lúc cũng tốt khi biết quad approx tính toán như thế nào. Cụ thể, ước lượng cũng có thể thất bại. Khi xảy ra, bạn sẽ thấy lỗi gì đó liên quan tới "Hessian". Sinh viên ngành lịch sử thế giới có thế biết Hessian là nhóm lính đánh thuê người Đức, thuê bởi nước Anh vào thế kỷ 18, để làm nhiều thứ bao gồm chiến đấu với quân kháng chiến Mỹ George Washington. Nhóm lính này được đặt tên theo vùng đất nay được gọi là trung tâm Đức, Hesse.

Hessian mà chúng ta quan tâm không có liên quan gì hết với lính đánh thuê. Nó được đặt tên theo nhà toán học Ludwig Otto Hesse (1811 - 1874). Một **Hessian** là ma trận vuông chứa đạo hàm bậc 2. Nó dùng nhiều trong toán học với nhiều mục đích, nhưng trong quad approx nó là đạo hàm bậc 2 của log_posterior sau khi cập nhật parameter. Và đạo hàm này có thể mô tả một phân phối Gaussian, vì log của gaussian là parabola. Parabola chỉ có đạo hàm bậc 2, cho nên khi chúng ta biết được đỉnh của parabola và đạo hàm bậc 2, ta biết mọi thứ về nó. Thực vậy đạo hàm bậc 2 của log của phân phối Gaussian tỉ lệ thuận với độ lệch chuẩn bình phương đảo ngược. Và biết độ lệch chuẩn là biết được tất cả về hình dạng của nó.

Độ lệch chuẩn này tính từ Hessian, cho nên tính Hessian là bắt buộc. Nhưng đôi khi phép tính bị lỗi, và golem bị mắc nghẹn khi tính Hessian. Lúc ấy, bạn có vài lựa chọn. Hi vọng vẫn còn đó. Nhưng bây giờ nó là đủ để ta hiểu thuật ngữ này và nó là một cách để tìm độ lệch chuẩn của quad approx.

---

### 2.4.5 Markov chain Monte Carlo.

Có rất nhiều dạng model quan trọng, như mixed-effects model, với grid approx hay quad approx là không thoả mãn. Những model này có tới hàng trăm hay hàng ngàn parameters. Grid approx không dùng được, vì nó tốn rất nhiều thời gian - Mặt trời đã lặn khi bạn tính xong grid. Quad approx có thể dùng được, nếu mọi thứ đều theo mẫu. Nhưng thông thường, sẽ có cái gì đó không đúng. Hơn nữa, model đa tầng không cho phép ta viết một hàm độc nhất để tính phân phối posterior. Có nghĩa là hàm để tìm maximum là không có, phải tính từng phần một.

Kết quả là, rất nhiều kỹ thuật fit model xuất hiện. Nổi tiếng nhất là **MARKOV CHAIN MONTE CARLO (MCMC)**, là một nhóm các động cơ cập nhật model phức tạp. Có thể nói MCMC là nguyên nhân bùng phát phân tích data Bayesian bắt đầu vào năm 1990. Mặc dù MCMC có từ lâu trước năm 1990, máy tính thì không theo kịp, nên ta phải cám ơn các kỹ sư máy tính.

Kỹ thuật MCMC khác ở chỗ là thay vì tính trực tiếp hay ước lượng phân phối posterior, nó chỉ đơn thuần là lấy mẫu từ posterior. Bạn sẽ có được một tập hợp các giá trị parameter, và tần suất của nó tương ứng với posterior. Bạn dùng tập hợp này để vẽ lên histogram của posterior.

Chúng ta làm việc luôn với tập hợp này, thay vì tạo công thức toán học của posterior bằng ước lượng. Và cách làm này tiện lợi hơn so với có posterior, bởi vì nó dễ hiểu hơn.

---

**Nghĩ nhiều hơn: Monte Carlo của ví dụ tung quả cầu.** 

```python
n_samples = 1000
p = [np.nan] * n_samples
p[0] = 0.5
W = 6
L = 3
with numpyro.handlers.seed(rng_key=0):
    for i in range(1, n_samples):
        p_new = numpyro.sample("p_new", dist.Normal(p[i - 1], 0.1))
        p_new = np.abs(p_new) if p_new < 0 else p_new
        p_new = 2 - p_new if p_new > 1 else p_new
        q0 = np.exp(dist.Binomial(W + L, p[i - 1]).log_prob(W))
        q1 = np.exp(dist.Binomial(W + L, p_new).log_prob(W))
        u = numpyro.sample("u", dist.Uniform())
        p[i] = p_new if u < q1 / q0 else p[i - 1]
```

```python
x = np.linspace(0, 1, 101)
az.plot_density({"p": p}, credible_interval=1)
plt.plot(x, np.exp(dist.Beta(W + 1, L + 1).log_prob(x)), "--");
```

<svg height="353.99952pt" version="1.1" viewBox="0 0 526.79952 353.99952" width="526.79952pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style></defs><g id="figure_1"><g id="patch_1"><path d="M 0 353.99952 
L 526.79952 353.99952 
L 526.79952 0 
L 0 0 
z
" style="fill:#ffffff;"></path></g><g id="axes_1"><g id="patch_2"><path d="M 7.2 329.750145 
L 519.59952 329.750145 
L 519.59952 25.3575 
L 7.2 25.3575 
z
" style="fill:#eeeeee;"></path></g><g id="matplotlib.axis_1"><g id="xtick_1"><g id="line2d_1"><path clip-path="url(#p125c6c25b9)" d="M 30.490887 329.750145 
L 30.490887 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_2"></g><g id="text_1"><!-- 0.0 --><defs><path d="M 31.78125 66.40625 
Q 24.171875 66.40625 20.328125 58.90625 
Q 16.5 51.421875 16.5 36.375 
Q 16.5 21.390625 20.328125 13.890625 
Q 24.171875 6.390625 31.78125 6.390625 
Q 39.453125 6.390625 43.28125 13.890625 
Q 47.125 21.390625 47.125 36.375 
Q 47.125 51.421875 43.28125 58.90625 
Q 39.453125 66.40625 31.78125 66.40625 
z
M 31.78125 74.21875 
Q 44.046875 74.21875 50.515625 64.515625 
Q 56.984375 54.828125 56.984375 36.375 
Q 56.984375 17.96875 50.515625 8.265625 
Q 44.046875 -1.421875 31.78125 -1.421875 
Q 19.53125 -1.421875 13.0625 8.265625 
Q 6.59375 17.96875 6.59375 36.375 
Q 6.59375 54.828125 13.0625 64.515625 
Q 19.53125 74.21875 31.78125 74.21875 
z
" id="DejaVuSans-48"></path><path d="M 10.6875 12.40625 
L 21 12.40625 
L 21 0 
L 10.6875 0 
z
" id="DejaVuSans-46"></path></defs><g style="fill:#262626;" transform="translate(19.3587 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_2"><g id="line2d_3"><path clip-path="url(#p125c6c25b9)" d="M 123.654436 329.750145 
L 123.654436 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_4"></g><g id="text_2"><!-- 0.2 --><defs><path d="M 19.1875 8.296875 
L 53.609375 8.296875 
L 53.609375 0 
L 7.328125 0 
L 7.328125 8.296875 
Q 12.9375 14.109375 22.625 23.890625 
Q 32.328125 33.6875 34.8125 36.53125 
Q 39.546875 41.84375 41.421875 45.53125 
Q 43.3125 49.21875 43.3125 52.78125 
Q 43.3125 58.59375 39.234375 62.25 
Q 35.15625 65.921875 28.609375 65.921875 
Q 23.96875 65.921875 18.8125 64.3125 
Q 13.671875 62.703125 7.8125 59.421875 
L 7.8125 69.390625 
Q 13.765625 71.78125 18.9375 73 
Q 24.125 74.21875 28.421875 74.21875 
Q 39.75 74.21875 46.484375 68.546875 
Q 53.21875 62.890625 53.21875 53.421875 
Q 53.21875 48.921875 51.53125 44.890625 
Q 49.859375 40.875 45.40625 35.40625 
Q 44.1875 33.984375 37.640625 27.21875 
Q 31.109375 20.453125 19.1875 8.296875 
z
" id="DejaVuSans-50"></path></defs><g style="fill:#262626;" transform="translate(112.522249 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="xtick_3"><g id="line2d_5"><path clip-path="url(#p125c6c25b9)" d="M 216.817985 329.750145 
L 216.817985 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_6"></g><g id="text_3"><!-- 0.4 --><defs><path d="M 37.796875 64.3125 
L 12.890625 25.390625 
L 37.796875 25.390625 
z
M 35.203125 72.90625 
L 47.609375 72.90625 
L 47.609375 25.390625 
L 58.015625 25.390625 
L 58.015625 17.1875 
L 47.609375 17.1875 
L 47.609375 0 
L 37.796875 0 
L 37.796875 17.1875 
L 4.890625 17.1875 
L 4.890625 26.703125 
z
" id="DejaVuSans-52"></path></defs><g style="fill:#262626;" transform="translate(205.685798 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="xtick_4"><g id="line2d_7"><path clip-path="url(#p125c6c25b9)" d="M 309.981535 329.750145 
L 309.981535 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_8"></g><g id="text_4"><!-- 0.6 --><defs><path d="M 33.015625 40.375 
Q 26.375 40.375 22.484375 35.828125 
Q 18.609375 31.296875 18.609375 23.390625 
Q 18.609375 15.53125 22.484375 10.953125 
Q 26.375 6.390625 33.015625 6.390625 
Q 39.65625 6.390625 43.53125 10.953125 
Q 47.40625 15.53125 47.40625 23.390625 
Q 47.40625 31.296875 43.53125 35.828125 
Q 39.65625 40.375 33.015625 40.375 
z
M 52.59375 71.296875 
L 52.59375 62.3125 
Q 48.875 64.0625 45.09375 64.984375 
Q 41.3125 65.921875 37.59375 65.921875 
Q 27.828125 65.921875 22.671875 59.328125 
Q 17.53125 52.734375 16.796875 39.40625 
Q 19.671875 43.65625 24.015625 45.921875 
Q 28.375 48.1875 33.59375 48.1875 
Q 44.578125 48.1875 50.953125 41.515625 
Q 57.328125 34.859375 57.328125 23.390625 
Q 57.328125 12.15625 50.6875 5.359375 
Q 44.046875 -1.421875 33.015625 -1.421875 
Q 20.359375 -1.421875 13.671875 8.265625 
Q 6.984375 17.96875 6.984375 36.375 
Q 6.984375 53.65625 15.1875 63.9375 
Q 23.390625 74.21875 37.203125 74.21875 
Q 40.921875 74.21875 44.703125 73.484375 
Q 48.484375 72.75 52.59375 71.296875 
z
" id="DejaVuSans-54"></path></defs><g style="fill:#262626;" transform="translate(298.849347 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-54"></use></g></g></g><g id="xtick_5"><g id="line2d_9"><path clip-path="url(#p125c6c25b9)" d="M 403.145084 329.750145 
L 403.145084 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_10"></g><g id="text_5"><!-- 0.8 --><defs><path d="M 31.78125 34.625 
Q 24.75 34.625 20.71875 30.859375 
Q 16.703125 27.09375 16.703125 20.515625 
Q 16.703125 13.921875 20.71875 10.15625 
Q 24.75 6.390625 31.78125 6.390625 
Q 38.8125 6.390625 42.859375 10.171875 
Q 46.921875 13.96875 46.921875 20.515625 
Q 46.921875 27.09375 42.890625 30.859375 
Q 38.875 34.625 31.78125 34.625 
z
M 21.921875 38.8125 
Q 15.578125 40.375 12.03125 44.71875 
Q 8.5 49.078125 8.5 55.328125 
Q 8.5 64.0625 14.71875 69.140625 
Q 20.953125 74.21875 31.78125 74.21875 
Q 42.671875 74.21875 48.875 69.140625 
Q 55.078125 64.0625 55.078125 55.328125 
Q 55.078125 49.078125 51.53125 44.71875 
Q 48 40.375 41.703125 38.8125 
Q 48.828125 37.15625 52.796875 32.3125 
Q 56.78125 27.484375 56.78125 20.515625 
Q 56.78125 9.90625 50.3125 4.234375 
Q 43.84375 -1.421875 31.78125 -1.421875 
Q 19.734375 -1.421875 13.25 4.234375 
Q 6.78125 9.90625 6.78125 20.515625 
Q 6.78125 27.484375 10.78125 32.3125 
Q 14.796875 37.15625 21.921875 38.8125 
z
M 18.3125 54.390625 
Q 18.3125 48.734375 21.84375 45.5625 
Q 25.390625 42.390625 31.78125 42.390625 
Q 38.140625 42.390625 41.71875 45.5625 
Q 45.3125 48.734375 45.3125 54.390625 
Q 45.3125 60.0625 41.71875 63.234375 
Q 38.140625 66.40625 31.78125 66.40625 
Q 25.390625 66.40625 21.84375 63.234375 
Q 18.3125 60.0625 18.3125 54.390625 
z
" id="DejaVuSans-56"></path></defs><g style="fill:#262626;" transform="translate(392.012896 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-56"></use></g></g></g><g id="xtick_6"><g id="line2d_11"><path clip-path="url(#p125c6c25b9)" d="M 496.308633 329.750145 
L 496.308633 25.3575 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_12"></g><g id="text_6"><!-- 1.0 --><defs><path d="M 12.40625 8.296875 
L 28.515625 8.296875 
L 28.515625 63.921875 
L 10.984375 60.40625 
L 10.984375 69.390625 
L 28.421875 72.90625 
L 38.28125 72.90625 
L 38.28125 8.296875 
L 54.390625 8.296875 
L 54.390625 0 
L 12.40625 0 
z
" id="DejaVuSans-49"></path></defs><g style="fill:#262626;" transform="translate(485.176445 343.887957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g></g><g id="matplotlib.axis_2"></g><g id="line2d_13"><path clip-path="url(#p125c6c25b9)" d="M 141.196767 312.689881 
L 146.245169 312.481965 
L 149.61077 312.052934 
L 152.976371 311.444519 
L 156.341972 310.630565 
L 161.390374 309.054269 
L 166.438775 307.160329 
L 173.169977 304.378373 
L 178.218379 302.0994 
L 183.266781 299.684569 
L 186.632382 297.908012 
L 191.680783 294.976651 
L 196.729185 291.729639 
L 205.143187 286.048501 
L 216.922791 277.773063 
L 220.288392 275.194279 
L 221.971193 273.823471 
L 225.336794 270.754202 
L 228.702395 267.094135 
L 230.385195 265.068091 
L 233.750796 260.474946 
L 237.116398 255.168727 
L 240.481999 249.215007 
L 243.8476 242.702633 
L 253.944403 222.836408 
L 258.992804 213.694363 
L 265.724006 201.442909 
L 269.089608 194.832006 
L 272.455209 187.526431 
L 275.82081 179.563538 
L 279.186411 171.082476 
L 301.062818 113.942231 
L 304.428419 106.202456 
L 307.79402 99.395884 
L 309.47682 96.44863 
L 311.159621 93.807487 
L 312.842421 91.466687 
L 314.525222 89.47675 
L 316.208022 87.746816 
L 317.890823 86.260089 
L 321.256424 83.782713 
L 324.622025 81.415162 
L 326.304825 80.06752 
L 327.987626 78.478459 
L 329.670427 76.709918 
L 333.036028 72.631295 
L 336.401629 67.866917 
L 341.45003 60.132637 
L 344.815631 55.127689 
L 348.181232 50.699603 
L 351.546833 46.7897 
L 354.912434 43.589207 
L 356.595235 42.225287 
L 358.278036 41.120124 
L 359.960836 40.154548 
L 361.643637 39.538425 
L 363.326437 39.207011 
L 365.009238 39.193529 
L 366.692038 39.492082 
L 368.374839 40.246807 
L 370.057639 41.441599 
L 371.74044 43.153464 
L 373.42324 45.212888 
L 375.106041 47.794449 
L 376.788841 50.878543 
L 380.154442 58.148591 
L 383.520043 66.765374 
L 386.885644 76.657748 
L 390.251246 87.468573 
L 395.299647 104.650266 
L 400.348049 121.761745 
L 403.71365 132.165687 
L 405.39645 136.888358 
L 407.079251 141.185598 
L 408.762051 145.149484 
L 410.444852 148.646373 
L 412.127652 151.71812 
L 413.810453 154.282212 
L 415.493253 156.464538 
L 417.176054 158.407968 
L 423.907256 165.271041 
L 427.272857 169.33248 
L 428.955658 171.78782 
L 430.638458 174.609905 
L 432.321259 177.743918 
L 435.68686 185.214372 
L 439.052461 193.817202 
L 442.418062 203.317684 
L 449.149264 223.35967 
L 452.514865 233.307004 
L 457.563267 247.110712 
L 460.928868 255.239486 
L 462.611668 259.011323 
L 464.294469 262.429814 
L 465.977269 265.499581 
L 467.66007 268.158935 
L 469.34287 270.443846 
L 471.025671 272.279637 
L 472.708471 273.693231 
L 474.391272 274.700217 
L 476.074072 275.17724 
L 476.074072 275.17724 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_14"><path clip-path="url(#p125c6c25b9)" d="M 141.196767 315.538989 
L 141.196767 312.689881 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_15"><path clip-path="url(#p125c6c25b9)" d="M 476.074072 315.914116 
L 476.074072 275.17724 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_16"><defs><path d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
" id="m02e061dc7e" style="stroke:#000000;"></path></defs><g clip-path="url(#p125c6c25b9)"><use style="fill:#2a2eec;stroke:#000000;" x="345.71626" xlink:href="#m02e061dc7e" y="315.51078"></use></g></g><g id="line2d_17"><path clip-path="url(#p125c6c25b9)" d="M 30.490887 315.51078 
L 35.149065 315.51078 
L 39.807242 315.510776 
L 44.465419 315.510729 
L 49.123597 315.510498 
L 53.781773 315.509738 
L 58.439951 315.507766 
L 63.09813 315.503419 
L 67.756306 315.494901 
L 72.414483 315.479626 
L 77.072659 315.454069 
L 81.730839 315.413625 
L 86.389015 315.352482 
L 91.047192 315.263518 
L 95.705372 315.138214 
L 100.363545 314.966598 
L 105.021725 314.737214 
L 109.679905 314.437117 
L 114.338078 314.051911 
L 118.996258 313.565793 
L 123.654431 312.961664 
L 128.312611 312.221224 
L 132.970791 311.325161 
L 137.628964 310.253282 
L 142.287144 308.984768 
L 146.945324 307.498363 
L 151.603497 305.772666 
L 156.26167 303.786375 
L 160.919857 301.518595 
L 165.57803 298.94915 
L 170.236203 296.058897 
L 174.894389 292.830022 
L 179.552562 289.24644 
L 184.210735 285.294014 
L 188.868922 280.961005 
L 193.527095 276.238307 
L 198.185268 271.119803 
L 202.843455 265.602594 
L 207.501628 259.687325 
L 212.159801 253.378568 
L 216.817974 246.684591 
L 221.476161 239.618212 
L 226.134334 232.196455 
L 230.792507 224.44083 
L 235.450694 216.377344 
L 240.108867 208.036857 
L 244.76704 199.45466 
L 249.425227 190.670552 
L 254.0834 181.728947 
L 258.741573 172.678433 
L 263.39976 163.571547 
L 268.057933 154.465005 
L 272.716106 145.418873 
L 277.374279 136.496102 
L 282.032452 127.762535 
L 286.690653 119.286091 
L 291.348826 111.136564 
L 296.006999 103.384588 
L 300.665172 96.101151 
L 305.323345 89.357196 
L 309.981518 83.222496 
L 314.639719 77.765079 
L 319.297892 73.050281 
L 323.956065 69.140478 
L 328.614238 66.093187 
L 333.272411 63.960961 
L 337.930584 62.790431 
L 342.588757 62.62143 
L 347.246958 63.485997 
L 351.905131 65.407447 
L 356.563304 68.399957 
L 361.221477 72.467368 
L 365.87965 77.602901 
L 370.537823 83.788143 
L 375.196023 90.993306 
L 379.854196 99.175401 
L 384.512369 108.279515 
L 389.170542 118.237686 
L 393.828715 128.96921 
L 398.486888 140.380867 
L 403.145061 152.367157 
L 407.803262 164.811585 
L 412.461435 177.586597 
L 417.119608 190.555317 
L 421.777781 203.573242 
L 426.435954 216.489786 
L 431.094127 229.150302 
L 435.752328 241.399305 
L 440.410501 253.082751 
L 445.068674 264.05216 
L 449.726847 274.168432 
L 454.38502 283.306296 
L 459.043193 291.359661 
L 463.701394 298.247166 
L 468.359567 303.918713 
L 473.01774 308.362662 
L 477.675913 311.613623 
L 482.334086 313.761194 
L 486.992259 314.959479 
L 491.650432 315.437539 
L 496.308633 315.51078 
" style="fill:none;stroke:#2a2eec;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;"></path></g><g id="patch_3"><path d="M 7.2 329.750145 
L 519.59952 329.750145 
" style="fill:none;"></path></g><g id="text_7"><!-- p --><defs><path d="M 18.109375 8.203125 
L 18.109375 -20.796875 
L 9.078125 -20.796875 
L 9.078125 54.6875 
L 18.109375 54.6875 
L 18.109375 46.390625 
Q 20.953125 51.265625 25.265625 53.625 
Q 29.59375 56 35.59375 56 
Q 45.5625 56 51.78125 48.09375 
Q 58.015625 40.1875 58.015625 27.296875 
Q 58.015625 14.40625 51.78125 6.484375 
Q 45.5625 -1.421875 35.59375 -1.421875 
Q 29.59375 -1.421875 25.265625 0.953125 
Q 20.953125 3.328125 18.109375 8.203125 
z
M 48.6875 27.296875 
Q 48.6875 37.203125 44.609375 42.84375 
Q 40.53125 48.484375 33.40625 48.484375 
Q 26.265625 48.484375 22.1875 42.84375 
Q 18.109375 37.203125 18.109375 27.296875 
Q 18.109375 17.390625 22.1875 11.75 
Q 26.265625 6.109375 33.40625 6.109375 
Q 40.53125 6.109375 44.609375 11.75 
Q 48.6875 17.390625 48.6875 27.296875 
z
" id="DejaVuSans-112"></path></defs><g style="fill:#262626;" transform="translate(258.32101 19.3575)scale(0.16 -0.16)"><use xlink:href="#DejaVuSans-112"></use></g></g></g></g><defs><clipPath id="p125c6c25b9"><rect height="304.392645" width="512.39952" x="7.2" y="25.3575"></rect></clipPath></defs></svg>

Thuật toán này gọi là **METROPOLIS**. Nó khá lạ, nhưng nó vẫn đúng.

---

## <center>2.5 Tổng kết</center><a name="2.5"></a>

Chương này giới thiệu các khái niệm và cơ chế hoạt động của Phân tích dữ liệu Bayesian. Mục đích là để tìm phân phối posterior. Posterior là số đếm các cách tương đối của mỗi khả năng của prior để tạo ra data. Và chúng được cập nhật kiểu Bayesian khi có data mới.

Trong cỗ máy Bayesian có tập hợp các variables và phân phối của variables đó. Xác suất của data, gọi là likelihood, là xác suất của data, khi đặt điều kiện là parameter của prior. Prior cho khả năng của từng giá trị của parameter có thể có được, trước khi cập nhật data. Thuyết xác suất cho phép chúng ta một phương pháp logic để tính xác suất, đó là Bayes' Theorem. Kết quả là phân phối posterior.

Trong đời thực, model Bayesian fit data bằng kỹ thuật ước lượng như Grid Approximation, Quadratic Approximation, Markov chain Monte Carlo. Mỗi kỹ thuật đều có điểm mạnh và yếu riêng.