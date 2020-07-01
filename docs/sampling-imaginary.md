---
title: "Chapter 3: Sampling the Imaginary"
description: "Chương 3: Lấy mẫu từ tưởng tượng"
---

> Bài viết dịch bởi người không chuyên, độc giả nào có góp ý xin phản hồi lại.

```python
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from jax import vmap
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist

%config InlineBackend.figure_formats = ["svg"]
az.style.use("arviz-darkgrid")
```
- [3.1 Lấy mẫu từ grid approx posterior](#3.1)
- [3.2 Lấy mẫu để mô tả](#3.2)
- [3.3 Lấy mẫu để mô phỏng dự đoán](#3.3)
- [3.4 Tổng kết](#3.4)
- [3.5 Bài tập](https://nbviewer.jupyter.org/github/vuongkienthanh/learn-bayes/blob/master/notebooks/chap3_ex.ipynb)


Nhiều sách giới thiệu thống kê Bayes và suy luận posterior bằng tình huống xét nghiệm y khoa. Ví dụ có một xét nghiệm vampire chính xác 95%, có thể mô tả bằng công thức Pr(dương \| vampire) = 0.95. Đây là một xét nghiệm có độ chính xác cao, hầu như xác định đúng vampire thật. Nhưng nó cũng có thể lỗi, dưới dạng dương tính giả. Một phần trăm trường hợp, nó chẩn đoán sai người thường là vampire, Pr(dương \| người) = 0.01. Thông tin cuối cùng là vampire rất hiếm, chiếm 0.1% dân số, suy ra Pr(vampire) = 0.001. Giả sử một người nào đó bị test dương tính. Vậy người đó có xác suất bao nhiêu phần trăm là con quái vật hút máu?

Cách tiếp cận đúng là dùng Bayes' theorem để đảo ngược lại xác suất:

$$ Pr(\text{vampire}| \text{dương} ) = \frac{Pr(\text{dương} |\text{vampire}) Pr(\text{vampire})} {Pr(\text{dương})} $$

Với Pr(dương) = trung bình xác suất của một kết quả test dương.

$$ Pr(\text{dương}) = Pr(\text{dương} |\text{vampire}) Pr(\text{vampire}) + Pr(\text{dương} |\text{người})(1 - Pr(\text{vampire})) $$

```python
Pr_Positive_Vampire = 0.95
Pr_Positive_Mortal = 0.01
Pr_Vampire = 0.001
numerator = Pr_Positive_Vampire * Pr_Vampire
Pr_Positive = tmp + Pr_Positive_Mortal * (1 - Pr_Vampire)
numerator / Pr_Positive
# 0.08683729
```

Chỉ có xác suất 8.7% đối tượng đó là vampire.

Nhiều người sẽ thấy rằng kết quả có vẻ ngược. Trước tiên, không có "Bayes" nào ở đây cả. Hãy nhớ rằng, Suy luận Bayes được phân biệt bởi tổng quan hoá xác suất, không phải là cách dùng công thức. Bởi vì tất cả xác suất tôi cung cấp là tần số của sự kiện, hơn là parameter lý thuyết, và mọi người đều đồng ý dùng Bayes' theorem ở ví dụ này. Thứ 2, và quan trọng hơn, ví dụ này làm cho suy luận Bayes trở nên khó hơn. Một số người thấy dễ hơn khi chỉ cần nhớ số nào để ở đâu, có lẽ họ không nắm được logic của quy trình. Nó là công thức trên trời rơi xuống. Nếu bạn thấy rắc rối, bởi vì bạn đang cố gắng hiểu nó.

Có một biện pháp giúp bạn dễ hiểu hơn. Giả sử, ta có:
1. Trong dân số 100,000 người, 100 người là vampire.
2. Trong 100 vampire đó, 95 trong số đó là có xét nghiệm dương tính.
3. Trong 99,000 người còn lại, 999 trong số đó là xét nghiệm dương tính.

Vậy số vampire trong tổng số xét nghiệm dương tính là:

$$ Pr(\text{vampire}|\text{dương}) = \frac{95}{95 + 999} = \frac{95}{1094} \approx 0.087 $$

Cách trình bày sử dụng số đếm thay vì xác suất, gọi là *format tần số* hoặc *tần số tự nhiên.* Tại sao format tần số giúp người ta dễ hiểu hơn? Một số người nghĩ rằng hệ thống thần kinh của loài người thích nghi tốt hơn khi nó nhận thông tin dưới dạng mà người bình thường nhận được. Trong thế giới tự nhiên, chúng ta chỉ gặp số đếm. Không ai đã từng gặp xác suất.

Trong chương này, ta sẽ lợi dụng yếu tố này, lấy mẫu từ các phân phối xác suất và tạo ra số đếm. Posterior cũng là một phân phối xác suất. Và cũng giống như các phân phối xác suất khác, ta có thể lấy mẫu tưởng tượng ra từ nó. Kết quả của việc lấy mẫu này là các giá trị của parameter. Phần lớn các parameter không có thực thể rõ ràng. Thống kê Bayesian xem phân phối parameter như các khả năng tương đối, chứ không phải một sự ngẫu nhiên trong đời thực. Posterior tạo ra tần suất mong đợi mà parameter khác nhau sẽ xuất hiện.

>**Nghĩ lại: hiện tượng tần suất tự nhiên tồn tại ở nhiều nơi.** Thay đổi cách trình bày của vấn đề thường làm nó dễ hơn để nhận biết và khơi dậy những ý tưởng mới mà không thể gặp ở dạng trình bày cũ. Trong vật lý, thay đổi cơ chế Newtonian và Lagranian có thể giúp giải quyết vấn đề đơn giản hơn. Trong sinh học tiến hoá, thay đổi inclusive fitness và multilevel selection có ánh sáng mới vào model cũ. Thống kê Bayes và non-Bayes cũng như thế.

Chương này dạy chúng ta những kỹ năng cơ bản để làm việc với mẫu của phân phối posterior. Có vẻ hơi sai khi bây giờ đã làm việc với mẫu, bởi vì posterior của ví dụ tung quả rất đơn giản. Nó đơn giản đến mức ta có thể tính nó bằng grid approximation hoặc dùng phương trình toán học. Nhưng có 2 lý do để ta phải học cách lấy mẫu từ sớm.

Một, nhiều nhà nghiên cứu không quen với tích phân, mặc dù họ hiểu cách để tổng quan data. Làm việc với mẫu chuyển biến vấn đề ở tích phân thành vấn đề tổng quan data, thành format tần số quen thuộc. Tích phân trong Bayes đồng nghĩa với tổng các xác suất trong khoảng nhất định. Nó có thể là một thách thức lớn. Nhưng khi ta có mẫu từ phân phối xác suất, nó chỉ là vấn đề của phép đếm tần số trong khoảng. Bằng phương pháp này, nhà nghiên cứu có thể hỏi và trả lời nhiều nghi vấn trong model, mà không cần đến nhà toán học. Vì lý do này, việc lấy mẫu từ posterior giúp ta có cảm giác trực quan hơn so với làm việc trực tiếp với xác suất và tích phân.

Hai, những phương pháp tính posterior mạnh nhất chỉ tạo được mẫu. Đa số các phương pháp này là biến thể của Markov chain Monte Carlo (MCMC). Cho nên nếu bạn quen với khái niệm và cách xử lý mẫu từ posterior, khi bạn cần fit model bằng MCMC, bạn phải hiểu là đang làm gì với kết quả từ MCMC. Tới chương 9, bạn sẽ dùng MCMC để tạo các loại và tăng độ phức tạp của model lên. MCMC không còn lại phương pháp cho chuyên gia, mà là một đồ nghề tiêu chuẩn cho khoa học định lượng.

>**Nghĩ lại: Thống kê không cứu được khoa học dỏm.** Ví dụ vampire này có cấu trúc giống như nhiều bài toán *phát hiện tín hiệu* khác: (1) Có một trạng thái nhị phân bị ẩn; (2) ta quan sát được một dấu hiệu không hoàn mỹ của trạng thái ẩn; (3) ta nên sử dụng Bayes' theorem để đưa hiệu ứng của dấu hiệu đó vào tính bất định.  
Suy luận thống kê cũng theo quy trình tương tự: (1) Một giả thuyết là đúng hoặc sai; (2) ta dùng dấu hiệu thống kê để chứng minh giả thuyết sai; (3) ta nên dùng Bayes' theorem để đưa dấu hiệu đó vào trạng thái của giả thuyết. Bước thứ 3 thường ít được thực hiện. Tôi không thích quy trình này lắm. Nhưng hãy xem ví dụ này, bạn sẽ thấy được các mối quan hệ. Giả sử ta có xác suất của một dấu hiệu dương tính, khi mà giả thuyết đúng, Pr(sig\|true) = 0.95. Đây là *power* của test. Giả sử ta có xác suất của một dấu hiệu dương tính, khi mà giả thuyết sai, Pr(sig\|false) = 0.05. Đây là xác suất dương tính giả, giống như phép thử mức độ ý nghĩa 5%. Cuối cùng, ta có xác suất nền khi giả thuyết đúng. Giả sử 1 trong 100 giả thuyết là đúng. Pr(true) = 0.01.  Không ai biết giả trị này, nhưng trong lịch sử khoa học thì số này rất nhỏ. Hãy tính posterior.  
$$ \text{Pr(true|pos)} = \frac{\text{Pr(pos|true)Pr(true)}}{\text{Pr(pos)}} = \frac{\text{Pr(pos|true)Pr(true)}}{\text{Pr(pos|true)Pr(true)} + \text{Pr(pos|false)Pr(false)}} $$  
Sau khi thay bằng các con số, ta có Pr(true\|pos) = 0.16. Và với dấu hiệu dương tính thì có 16% cơ hội giả thuyết là đúng. Đây giống như hiện tượng xác suất nền thấp trong ví dụ vampire. Bạn có thể cho tỉ lệ dương tính giả thấp xuống 1% và nâng xác suất posterior lên 0.5, và như một đồng xu. Quan trọng nhất ở đây là xác suất nền, và nó cần đặt nhiều \\\\suy nghĩ, chứ không phải kiểm tra nó.

## <center>3.1 Lấy mẫu từ grid approx posterior</center><a name="3.1"></a>

Trước khi làm việc với mẫu, ta phải tạo được nó. Ta nhắc lại phương pháp grid approx để tìm posterior trong ví dụ tung quả cầu.

```python
p_grid = np.linspace(start=0, stop=1, num=1000)
prob_p = np.repeat(1, 1000)
prob_data = np.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6))
posterior = prob_data * prob_p
posterior = posterior / np.sum(posterior)
```

Giờ ta muốn lấy 10,000 mẫu từ posterior này. Tưởng tượng posterior như một rổ chứa đầy các giá trị của parameter, các con số như 0.1, 0.7, 0.5, 1,.. Trong cái rổ, mỗi giá trị tồn tại tỉ lệ thuận với xác suất posterior, ví dụ như giá trị gần đỉnh thì thường gặp hơn các giá trị ở hai đuôi. Ta lấy 10,000 giá trị từ trong rổ. Cho rằng cái rổ đã trộn đều, các mẫu lấy từ nó sẽ có tỉ lệ thành phần giống hoàn toàn với mật độ posterior. Cho nên mỗi giá trị *p* sẽ xuất hiện trong mẫu sẽ giống với xác suất posterior của mỗi giá trị.

```python
samples = p_grid[dist.Categorical(probs=posterior).sample(PRNGKey(0), (10000,))]
```

`dist.Categorical` xem các giá trị trong posterior từ grid approx có xác suất như nhau, sau đó lấy mẫu lặp lại 10,000 lần. Ta có thể plot kết quả này như sau:

```python
plt.scatter(range(len(samples)), samples, alpha=0.2);
```

<svg height="248.889375pt" version="1.1" viewBox="0 0 382.04511 248.889375" width="382.04511pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style></defs><g id="figure_1"><g id="patch_1"><path d="M 0 248.889375 
L 382.04511 248.889375 
L 382.04511 0 
L 0 0 
z
" style="fill:none;"></path></g><g id="axes_1"><g id="patch_2"><path d="M 32.964375 224.64 
L 367.764375 224.64 
L 367.764375 7.2 
L 32.964375 7.2 
z
" style="fill:#eeeeee;"></path></g><g id="matplotlib.axis_1"><g id="xtick_1"><g id="line2d_1"><path clip-path="url(#p8418683863)" d="M 48.18283 224.64 
L 48.18283 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_2"></g><g id="text_1"><!-- 0 --><defs><path d="M 31.78125 66.40625 
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
" id="DejaVuSans-48"></path></defs><g style="fill:#262626;" transform="translate(43.72908 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_2"><g id="line2d_3"><path clip-path="url(#p8418683863)" d="M 109.061536 224.64 
L 109.061536 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_4"></g><g id="text_2"><!-- 2000 --><defs><path d="M 19.1875 8.296875 
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
" id="DejaVuSans-50"></path></defs><g style="fill:#262626;" transform="translate(91.246536 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-50"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-48"></use><use x="190.869141" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_3"><g id="line2d_5"><path clip-path="url(#p8418683863)" d="M 169.940242 224.64 
L 169.940242 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_6"></g><g id="text_3"><!-- 4000 --><defs><path d="M 37.796875 64.3125 
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
" id="DejaVuSans-52"></path></defs><g style="fill:#262626;" transform="translate(152.125242 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-52"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-48"></use><use x="190.869141" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_4"><g id="line2d_7"><path clip-path="url(#p8418683863)" d="M 230.818948 224.64 
L 230.818948 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_8"></g><g id="text_4"><!-- 6000 --><defs><path d="M 33.015625 40.375 
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
" id="DejaVuSans-54"></path></defs><g style="fill:#262626;" transform="translate(213.003948 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-54"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-48"></use><use x="190.869141" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_5"><g id="line2d_9"><path clip-path="url(#p8418683863)" d="M 291.697654 224.64 
L 291.697654 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_10"></g><g id="text_5"><!-- 8000 --><defs><path d="M 31.78125 34.625 
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
" id="DejaVuSans-56"></path></defs><g style="fill:#262626;" transform="translate(273.882654 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-56"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-48"></use><use x="190.869141" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_6"><g id="line2d_11"><path clip-path="url(#p8418683863)" d="M 352.57636 224.64 
L 352.57636 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_12"></g><g id="text_6"><!-- 10000 --><defs><path d="M 12.40625 8.296875 
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
" id="DejaVuSans-49"></path></defs><g style="fill:#262626;" transform="translate(330.30761 238.777812)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use><use x="63.623047" xlink:href="#DejaVuSans-48"></use><use x="127.246094" xlink:href="#DejaVuSans-48"></use><use x="190.869141" xlink:href="#DejaVuSans-48"></use><use x="254.492188" xlink:href="#DejaVuSans-48"></use></g></g></g></g><g id="matplotlib.axis_2"><g id="ytick_1"><g id="line2d_13"><path clip-path="url(#p8418683863)" d="M 32.964375 200.34269 
L 367.764375 200.34269 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_14"></g><g id="text_7"><!-- 0.2 --><defs><path d="M 10.6875 12.40625 
L 21 12.40625 
L 21 0 
L 10.6875 0 
z
" id="DejaVuSans-46"></path></defs><g style="fill:#262626;" transform="translate(7.2 205.661596)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="ytick_2"><g id="line2d_15"><path clip-path="url(#p8418683863)" d="M 32.964375 153.579028 
L 367.764375 153.579028 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_16"></g><g id="text_8"><!-- 0.4 --><g style="fill:#262626;" transform="translate(7.2 158.897934)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="ytick_3"><g id="line2d_17"><path clip-path="url(#p8418683863)" d="M 32.964375 106.815365 
L 367.764375 106.815365 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_18"></g><g id="text_9"><!-- 0.6 --><g style="fill:#262626;" transform="translate(7.2 112.134271)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-54"></use></g></g></g><g id="ytick_4"><g id="line2d_19"><path clip-path="url(#p8418683863)" d="M 32.964375 60.051702 
L 367.764375 60.051702 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_20"></g><g id="text_10"><!-- 0.8 --><g style="fill:#262626;" transform="translate(7.2 65.370609)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-56"></use></g></g></g><g id="ytick_5"><g id="line2d_21"><path clip-path="url(#p8418683863)" d="M 32.964375 13.28804 
L 367.764375 13.28804 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_22"></g><g id="text_11"><!-- 1.0 --><g style="fill:#262626;" transform="translate(7.2 18.606946)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g></g><g id="PathCollection_1"><defs><path d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
L 32.964375 7.2 
" style="fill:none;"></path></g><g id="patch_4"><path d="M 367.764375 224.64 
L 367.764375 7.2 
" style="fill:none;"></path></g><g id="patch_5"><path d="M 32.964375 224.64 
L 367.764375 224.64 
" style="fill:none;"></path></g><g id="patch_6"><path d="M 32.964375 7.2 
L 367.764375 7.2 
" style="fill:none;"></path></g></g></g><defs><clipPath id="p8418683863"><rect height="217.44" width="334.8" x="32.964375" y="7.2"></rect></clipPath></defs></svg>

Trong biểu đồ trên, bạn sẽ thấy giống như mình đang bay trên phân phối posterior và nhìn xuống dưới. Có nhiều mẫu ở vùng dày đặc ở 0.6 và ít hơn ở vùng dưới 0.25.

```python
az.plot_density({"": samples}, credible_interval=1);
```

<svg height="296.39952pt" version="1.1" viewBox="0 0 440.399528 296.39952" width="440.399528pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style></defs><g id="figure_1"><g id="patch_1"><path d="M 0 296.39952 
L 440.399528 296.39952 
L 440.399528 0 
L 0 0 
z
" style="fill:none;"></path></g><g id="axes_1"><g id="patch_2"><path d="M 7.2 272.150145 
L 427.13725 272.150145 
L 427.13725 7.2 
L 7.2 7.2 
z
" style="fill:#eeeeee;"></path></g><g id="matplotlib.axis_1"><g id="xtick_1"><g id="line2d_1"><path clip-path="url(#p0269339e2a)" d="M 48.623653 272.150145 
L 48.623653 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_2"></g><g id="text_1"><!-- 0.2 --><defs><path d="M 31.78125 66.40625 
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
" id="DejaVuSans-46"></path><path d="M 19.1875 8.296875 
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
" id="DejaVuSans-50"></path></defs><g style="fill:#262626;" transform="translate(37.491466 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use></g></g></g><g id="xtick_2"><g id="line2d_3"><path clip-path="url(#p0269339e2a)" d="M 95.304114 272.150145 
L 95.304114 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_4"></g><g id="text_2"><!-- 0.3 --><defs><path d="M 40.578125 39.3125 
Q 47.65625 37.796875 51.625 33 
Q 55.609375 28.21875 55.609375 21.1875 
Q 55.609375 10.40625 48.1875 4.484375 
Q 40.765625 -1.421875 27.09375 -1.421875 
Q 22.515625 -1.421875 17.65625 -0.515625 
Q 12.796875 0.390625 7.625 2.203125 
L 7.625 11.71875 
Q 11.71875 9.328125 16.59375 8.109375 
Q 21.484375 6.890625 26.8125 6.890625 
Q 36.078125 6.890625 40.9375 10.546875 
Q 45.796875 14.203125 45.796875 21.1875 
Q 45.796875 27.640625 41.28125 31.265625 
Q 36.765625 34.90625 28.71875 34.90625 
L 20.21875 34.90625 
L 20.21875 43.015625 
L 29.109375 43.015625 
Q 36.375 43.015625 40.234375 45.921875 
Q 44.09375 48.828125 44.09375 54.296875 
Q 44.09375 59.90625 40.109375 62.90625 
Q 36.140625 65.921875 28.71875 65.921875 
Q 24.65625 65.921875 20.015625 65.03125 
Q 15.375 64.15625 9.8125 62.3125 
L 9.8125 71.09375 
Q 15.4375 72.65625 20.34375 73.4375 
Q 25.25 74.21875 29.59375 74.21875 
Q 40.828125 74.21875 47.359375 69.109375 
Q 53.90625 64.015625 53.90625 55.328125 
Q 53.90625 49.265625 50.4375 45.09375 
Q 46.96875 40.921875 40.578125 39.3125 
z
" id="DejaVuSans-51"></path></defs><g style="fill:#262626;" transform="translate(84.171927 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-51"></use></g></g></g><g id="xtick_3"><g id="line2d_5"><path clip-path="url(#p0269339e2a)" d="M 141.984575 272.150145 
L 141.984575 7.2 
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
" id="DejaVuSans-52"></path></defs><g style="fill:#262626;" transform="translate(130.852388 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-52"></use></g></g></g><g id="xtick_4"><g id="line2d_7"><path clip-path="url(#p0269339e2a)" d="M 188.665036 272.150145 
L 188.665036 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_8"></g><g id="text_4"><!-- 0.5 --><defs><path d="M 10.796875 72.90625 
L 49.515625 72.90625 
L 49.515625 64.59375 
L 19.828125 64.59375 
L 19.828125 46.734375 
Q 21.96875 47.46875 24.109375 47.828125 
Q 26.265625 48.1875 28.421875 48.1875 
Q 40.625 48.1875 47.75 41.5 
Q 54.890625 34.8125 54.890625 23.390625 
Q 54.890625 11.625 47.5625 5.09375 
Q 40.234375 -1.421875 26.90625 -1.421875 
Q 22.3125 -1.421875 17.546875 -0.640625 
Q 12.796875 0.140625 7.71875 1.703125 
L 7.71875 11.625 
Q 12.109375 9.234375 16.796875 8.0625 
Q 21.484375 6.890625 26.703125 6.890625 
Q 35.15625 6.890625 40.078125 11.328125 
Q 45.015625 15.765625 45.015625 23.390625 
Q 45.015625 31 40.078125 35.4375 
Q 35.15625 39.890625 26.703125 39.890625 
Q 22.75 39.890625 18.8125 39.015625 
Q 14.890625 38.140625 10.796875 36.28125 
z
" id="DejaVuSans-53"></path></defs><g style="fill:#262626;" transform="translate(177.532849 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-53"></use></g></g></g><g id="xtick_5"><g id="line2d_9"><path clip-path="url(#p0269339e2a)" d="M 235.345497 272.150145 
L 235.345497 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_10"></g><g id="text_5"><!-- 0.6 --><defs><path d="M 33.015625 40.375 
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
" id="DejaVuSans-54"></path></defs><g style="fill:#262626;" transform="translate(224.21331 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-54"></use></g></g></g><g id="xtick_6"><g id="line2d_11"><path clip-path="url(#p0269339e2a)" d="M 282.025958 272.150145 
L 282.025958 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_12"></g><g id="text_6"><!-- 0.7 --><defs><path d="M 8.203125 72.90625 
L 55.078125 72.90625 
L 55.078125 68.703125 
L 28.609375 0 
L 18.3125 0 
L 43.21875 64.59375 
L 8.203125 64.59375 
z
" id="DejaVuSans-55"></path></defs><g style="fill:#262626;" transform="translate(270.893771 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-55"></use></g></g></g><g id="xtick_7"><g id="line2d_13"><path clip-path="url(#p0269339e2a)" d="M 328.706419 272.150145 
L 328.706419 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_14"></g><g id="text_7"><!-- 0.8 --><defs><path d="M 31.78125 34.625 
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
" id="DejaVuSans-56"></path></defs><g style="fill:#262626;" transform="translate(317.574231 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-56"></use></g></g></g><g id="xtick_8"><g id="line2d_15"><path clip-path="url(#p0269339e2a)" d="M 375.38688 272.150145 
L 375.38688 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_16"></g><g id="text_8"><!-- 0.9 --><defs><path d="M 10.984375 1.515625 
L 10.984375 10.5 
Q 14.703125 8.734375 18.5 7.8125 
Q 22.3125 6.890625 25.984375 6.890625 
Q 35.75 6.890625 40.890625 13.453125 
Q 46.046875 20.015625 46.78125 33.40625 
Q 43.953125 29.203125 39.59375 26.953125 
Q 35.25 24.703125 29.984375 24.703125 
Q 19.046875 24.703125 12.671875 31.3125 
Q 6.296875 37.9375 6.296875 49.421875 
Q 6.296875 60.640625 12.9375 67.421875 
Q 19.578125 74.21875 30.609375 74.21875 
Q 43.265625 74.21875 49.921875 64.515625 
Q 56.59375 54.828125 56.59375 36.375 
Q 56.59375 19.140625 48.40625 8.859375 
Q 40.234375 -1.421875 26.421875 -1.421875 
Q 22.703125 -1.421875 18.890625 -0.6875 
Q 15.09375 0.046875 10.984375 1.515625 
z
M 30.609375 32.421875 
Q 37.25 32.421875 41.125 36.953125 
Q 45.015625 41.5 45.015625 49.421875 
Q 45.015625 57.28125 41.125 61.84375 
Q 37.25 66.40625 30.609375 66.40625 
Q 23.96875 66.40625 20.09375 61.84375 
Q 16.21875 57.28125 16.21875 49.421875 
Q 16.21875 41.5 20.09375 36.953125 
Q 23.96875 32.421875 30.609375 32.421875 
z
" id="DejaVuSans-57"></path></defs><g style="fill:#262626;" transform="translate(364.254692 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-57"></use></g></g></g><g id="xtick_9"><g id="line2d_17"><path clip-path="url(#p0269339e2a)" d="M 422.067341 272.150145 
L 422.067341 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_18"></g><g id="text_9"><!-- 1.0 --><defs><path d="M 12.40625 8.296875 
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
" id="DejaVuSans-49"></path></defs><g style="fill:#262626;" transform="translate(410.935153 286.287957)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use></g></g></g></g><g id="matplotlib.axis_2"></g><g id="line2d_19"><path clip-path="url(#p0269339e2a)" d="M 26.288057 258.413347 
L 33.961647 258.307899 
L 41.635238 257.985367 
L 47.390431 257.525117 
L 53.145624 256.781485 
L 56.982419 256.06799 
L 60.819215 255.15252 
L 64.65601 254.038242 
L 68.492806 252.709061 
L 72.329601 251.192479 
L 76.166396 249.486562 
L 81.921589 246.651288 
L 87.676782 243.550091 
L 93.431975 240.221634 
L 99.187168 236.572811 
L 103.023964 233.916566 
L 106.860759 231.001304 
L 110.697554 227.743126 
L 114.53435 224.155276 
L 118.371145 220.157005 
L 122.20794 215.707101 
L 126.044736 210.785818 
L 129.881531 205.460553 
L 143.310315 186.023438 
L 147.14711 180.968703 
L 150.983905 176.229024 
L 160.575894 164.466695 
L 164.412689 159.174785 
L 168.249484 153.230958 
L 172.08628 146.535378 
L 175.923075 139.125177 
L 179.75987 131.228929 
L 189.351859 111.105588 
L 193.188654 103.588985 
L 197.025449 96.563382 
L 200.862245 89.991947 
L 206.617438 80.673672 
L 220.046221 59.695313 
L 223.883017 54.088184 
L 227.719812 48.99813 
L 229.63821 46.675619 
L 233.475005 42.613213 
L 235.393403 40.799264 
L 239.230198 37.607026 
L 243.066993 34.779021 
L 252.658982 28.185921 
L 258.414175 24.470713 
L 262.25097 22.331941 
L 264.169368 21.412345 
L 266.087765 20.643419 
L 268.006163 20.031875 
L 269.924561 19.554889 
L 271.842958 19.287771 
L 273.761356 19.243188 
L 275.679754 19.409443 
L 277.598151 19.826225 
L 279.516549 20.50069 
L 281.434947 21.46117 
L 283.353344 22.717659 
L 285.271742 24.299576 
L 287.19014 26.173123 
L 289.108537 28.369006 
L 291.026935 30.914406 
L 292.945333 33.688979 
L 294.86373 36.677112 
L 298.700526 43.309548 
L 304.455719 54.071134 
L 317.884502 79.866743 
L 321.721298 87.586257 
L 325.558093 95.668448 
L 331.313286 108.27267 
L 340.905274 129.979135 
L 346.660467 143.357856 
L 352.41566 157.331959 
L 367.762842 195.597701 
L 371.599637 204.329722 
L 375.436432 212.386146 
L 379.273228 219.688822 
L 383.110023 226.260119 
L 386.946818 232.102552 
L 390.783614 237.213803 
L 392.702011 239.482213 
L 394.620409 241.570308 
L 396.538807 243.416918 
L 398.457204 245.051318 
L 400.375602 246.431806 
L 402.294 247.570602 
L 404.212397 248.429385 
L 406.130795 249.0098 
L 408.049193 249.296449 
L 408.049193 249.296449 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_20"><path clip-path="url(#p0269339e2a)" d="M 26.288057 260.015788 
L 26.288057 258.413347 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_21"><path clip-path="url(#p0269339e2a)" d="M 408.049193 260.106957 
L 408.049193 249.296449 
" style="fill:none;stroke:#2a2eec;stroke-linecap:round;stroke-width:1.5;"></path></g><g id="line2d_22"><defs><path d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
" id="m37ee424fdb" style="stroke:#000000;"></path></defs><g clip-path="url(#p0269339e2a)"><use style="fill:#2a2eec;stroke:#000000;" x="252.868861" xlink:href="#m37ee424fdb" y="259.999922"></use></g></g><g id="patch_3"><path d="M 7.2 272.150145 
L 427.13725 272.150145 
" style="fill:none;"></path></g></g></g><defs><clipPath id="p0269339e2a"><rect height="264.950145" width="419.93725" x="7.2" y="7.2"></rect></clipPath></defs></svg>

Hình này cho thấy mật độ từ mẫu ta đã lấy, nó rất giống với posterior lý tưởng mà ta tính từ grid approx. Nếu bạn lấy nhiều mẫu hơn, 1e5 hoặc 1e6, mật độ này sẽ ngày càng giống với posterior lý tưởng.

Những gì bạn làm là tái lập lại mật độ posterior mà bạn đã tính. Nó không có ý nghĩa nhiều. Nhưng tiếp theo sau đây là sử dụng mẫu này để mô tả và hiểu thấu posterior, nó mới là có giá trị thực sự.

## <center>3.2 Lấy mẫu để mô tả</center><a name="3.2"></a>

Khi model đã tạo được phân phối posterior, nhiệm vụ của model đã xong. Nhưng công việc của bạn chỉ mới bắt đầu. Bạn cần phải mô tả và diễn giải phân phối posterior. Bằng cách nào thì tuỳ mục đích của bạn. Nhưng câu hỏi thường gặp gồm:

- Xác suất nằm nhỏ một giá trị nào đó là bao nhiêu?
- Xác suất nằm trong khoảng giá trị nào đó là bao nhiêu?
- Giá trị nào nằm ở 5% dưới của xác suất posterior?
- Khoảng giá trị nào chứa 90% của xác suất posterior?
- Giá trị nào có xác suất cao nhất?

Có thể chia làm 3 nhóm câu hỏi: (1) khoảng ranh giới xác định; (2) khoảng mật độ xác định; (3) ước lượng điểm.

### 3.2.1 Khoảng ranh giới xác định.

Câu hỏi là xác suất của tỉ lệ bề mặt nước < 0.5. Bằng posterior từ grid approx, bạn chỉ việc dùng tổng các xác suất mà giá trị parameter < 0.5

```python
np.sum(posterior[p_grid < 0.5])
# DeviceArray(0.17187457, dtype=float32)
```

Khoảng 17% của xác suất posterior là dưới 0.5. Không có gì dễ hơn. Nhưng bởi vì grid approx không được dùng trong thực tế, nó không dễ như vậy. Khi mà có nhiều hơn một parameter trong phân phối posterior, phép tính cộng đơn giản cũng không còn đơn giản nữa.

Vậy hãy xem cách nào để thực hiện phép tính này, bằng sử dụng mẫu lấy từ posterior. Cách làm này tổng quát hoá các model phức tạp với nhiều parameter, và bạn có thể dùng chúng ở mọi nơi. Những gì bạn cần là cộng hết những mẫu nào dưới 0.5, và chia nó cho tổng số lần lấy mẫu. Hay nói khác hơn, là tần suất của parameter < 0.5:

```python
np.sum(samples < 0.5) / 1e4
# DeviceArray(0.1711, dtype=float32)
```

Và kết quả này gần giống với kết quả từ grid approx, mặc dù có thể không giống chính xác với kết quả của bạn, bởi vì mẫu mà bạn rút từ posterior hơi khác. Bạn có thể làm tương tự với khoảng giá trị khác.

```python
np.sum((samples > 0.5) & (samples < 0.75)) / 1e4
# DeviceArray(0.6025, dtype=float32)
```

![](/assets/images/figure 3-2.png)

### 3.2.2 Khoảng mật độ xác định

Thường gặp hơn trong các tạp chí khoa học là báo cáo khoảng giá trị của mật độ cụ thể, gọi là **KHOẢNG TIN CẬY (CONFIDENCE INTERVAL)**. Một khoảng của xác suất posterior, có thể được gọi là **CREDIBLE INTERVAL**. Chứng ta sẽ gọi nó là **KHOẢNG PHÙ HỢP(COMPATIBILITY INTERVAL - CI)**, để tránh sự lầm tưởng của "confidence (tin cậy)" hay "credibility (tín nhiệm)". Khoảng này là một khoảng các giá trị của parameter mà phù hợp với model và data. Model và data không tạo ra sự tin vậy, và khoảng này cũng không.

Khoảng posterior gồm 2 giá trị của parameter mà giữa chúng chứ lượng xác suất định trước hay mật độ xác suất. Dạng câu hỏi này, ta sẽ dễ dàng hơn khi tìm câu trả lời bằng mẫu của posterior hơn là dùng grid approx. Giả sử bạn muốn biết khoảng giá trị chứa 80% dưới xác suất posterior. Bạn biết khoảng này bắt đầu từ 0, và bạn tiếp 80th percentile của nó:

```python
np.quantile(samples, 0.8)
# DeviceArray(0.7637638, dtype=float32)
```

Tương tự cũng giống như khoảng 10th và 90th:

```python
np.quantile(samples, [0.1, 0.9])
# DeviceArray([0.44644645, 0.81681681], dtype=float32)
```

![](/assets/images/figure 3-2b.png)

Khoảng tin cậy giống như vậy, với mật độ bằng nhau ở 2 đuôi rất thường gặp trong khoa học. Ta sẽ gọi nó là **PERCENTILE INTERVAL (PI)**. Những khoảng này cho kết quả rất tốt để biểu diễn hình dạng của phân phối, miễn là phân phối đừng có bất đối xứng quá. Nhưng trong suy luận parameter nào là phù hợp nhất với data, chúng không hoàn mỹ.

![](/assets/images/figure 3-3.png)

Trong ví dụ trên, posterior được tạo từ prior uniform và binomial với 3 lần tung quả cầu ra nước trong 3 lần tung. Nó lệch rất nhiều, có maximum ở biên giới *p* = 1. Khoảng 50% mật độ xác suất ở giữa là 25th và 75th percentile:

```python
p_grid = np.linspace(start=0, stop=1, num=1000)
prior = np.repeat(1, 1000)
likelihood = np.exp(dist.Binomial(total_count=3, probs=p_grid).log_prob(3))
posterior = likelihood * prior
posterior = posterior / np.sum(posterior)
samples = p_grid[
    dist.Categorical(probs=posterior).sample(PRNGKey(0), (10000,))]
np.percentile(samples, q=(25, 75))
# DeviceArray([0.7077077, 0.93193191], dtype=float32)
```

Khoảng này gán 25% mật độ trên và dưới khoảng, và 50% mật độ ở giữa. Nó bỏ qua giá trị có xác suất cao nhất gần *p* = 1. Cho nên để mô tả hình dáng của posterior - đây là những gì ta cần ở khoảng này - khoảng percentile có thể gây sai.

>**Nghĩ lại: Tại sao 95%?** Khoảng mật độ được dùng nhiều nhất trong khoa học tự nhiên và xã hội là 95%. Khoảng này cho 5% xác suất ở bên ngoài, tương ứng với 5% cơ hội parameter này không nằm trong trong khoảng. Truyền thống này phản ánh ngưỡng tin cậy 5% hay p<0.05. Nó chỉ mang ý nghĩa thuận tiện. Ronald Fisher thường bị đổ tội cho lựa chọn 95% này: *"Độ lệch chuẩn cho p=0.05 là 1.96 hay gần bằng 2; nó khá tiện khi chọn điểm này làm giới hạn để đánh giá một điểm khác là có tin cậy hay không."* Nhiều người không nghĩ rằng sự tiện lợi là một tiêu chí lựa chọn. Sau này trong sự nghiệp của ông, Fisher chủ động khuyên ngăn dùng ngưỡng này để kiểm tra mức tin cậy.  
Vậy bạn cần làm gì? Không có chuẩn mực nào cả, nhưng có suy nghĩ là tốt. Nếu bạn cố gắng chứng minh khoảng tin cậy không chứa một giá trị nào đó, bạn hãy dùng khoảng tin cậy rộng nhất không chứa giá trị đó có thể. Thông thường, khoảng tin cậy là một ánh xạ của hình dáng của phân phối. Bạn tốt hơn hết là có nhiều khoảng tin cậy khác nhau. Tại sao lại không trình bày khoảng 67%, 89%, 97%, kèm theo median? Tại sao lại là những giá trị này? Không lý do. Nó là số nguyên tố, dễ nhớ hơn. Những gì quan trọng là nó đủ rộng để minh hoạ cho hình dáng của posterior. Và các giá trị này tránh 95%, bởi con số 95% truyền thống này khuyến khích người đọc suy nghĩ nhiều hơn.

Ngược lại, hình bên phải là 50% **KHOẢNG MẬT ĐỘ LỚN NHẤT (HIGHEST POSTERIOR DENSITY INTERVAL - HDPI)**. HPDI là khoảng hẹp nhất chứa mật độ xác suất định trước. Nếu bạn nghĩ lại, nó có vô số các khoảng có cùng mật độ xác định. Nhưng nếu bạn muốn khoảng giá trị tốt nhất mà đúng với data, bạn cần khoảng dày đặc nhất. Nó là HPDI.

```python
numpyro.diagnostics.hpdi(samples, prob=0.5)
# array([0.8418418, 0.998999 ], dtype=float32)
```

HPDI cho ta khoảng tin cậy chứa xác suất lớn nhất, cũng như hẹp hơn rất nhiều. Chiều rộng 0.16 nhỏ hơn 0.23.

HPDI có lợi thế hơn mặc định khoảng giữa. Đa số, 2 khoảng này là gần bằng nhau. Nó khác nhau trong trường hợp này cho do phân phối posterior quá lệch. Nếu ta dùng 6W và 3L, 2 khoảng này trùng nhau. Nên nhớ rằng chúng ta không phải phóng tên lửa hay bắn hạt nhân, cho nên chọn precision số thập phân là 5 không giúp khoa học tiến bộ.

HPDI cũng có bất lợi. HPDI cần tính toán nhiều hơn khoảng giữa và bị ảnh hưởng bởi *biến thiên mô phỏng*, tức là mẫu càng nhiều thì HPDI càng nhạy cảm hơn. Và chính HPDI cũng khó hiểu hơn so với khoảng percentile đối với nhiều người nghiên cứu.

Nhìn chung, nếu việc lựa chọn khoảng tin cậy có giá trị lớn, thì bạn không nên dùng khoảng tin cậy để mô tả posterior. Nhớ rằng, toàn bộ posterior là "ước lượng Bayesian". Nó mô tả xác suất tương đối của mỗi giá trị có thể của parameter. Khoảng phân phối chỉ có ích để mô tả nó. Nếu khoảng tin cậy bạn chọn ảnh hưởng nhiều đến suy luận, thì bạn tốt hơn hết là vẽ toàn bộ posterior.

>**Nghĩ lại: Khoảng tin cậy là gì?** Người ta thường nói khoảng tin cậy 95% là xác suất 0.95 để giá trị thật nằm trong khoảng tin cậy. Trong suy luận thống kê non-Bayes, mệnh đề này là sai, vì suy luận non-Bayes không cho phép dùng xác suất để mô tả tính bất định của parameter. Ta phải nói, nếu ta lặp lại thí nghiệm này và phân tích với một số lượng lớn, thì 95% trong số lần đó sẽ ra khoảng tin cậy chứa giá trị thật. Nếu như bạn không rõ sự khác biệt này, thì bạn cũng giống như mọi người. Định nghĩa này khá là trừu tượng, và nhiều người đã diễn giải chúng thông qua cách của Bayesian.  
Nhưng nếu bạn là Bayes hay không, thì khoảng 95% này không phải lúc nào cũng chứa giá trị thật 95% các trường hợp. Khoa học đã chứng minh khoảng tin cậy đã làm cho con người tự tin thái quá. Con số này nằm trong thế giới nhỏ, nó chỉ đúng trong thế giới nhỏ của nó. Nên nó sẽ không bao giờ được áp dụng chính xác tuyệt đối trong thế giới thực. Nó là những gì golem tin tưởng, bạn được quyền tin vào những thứ khác. Dù sao, độ rộng của khoảng tin cậy, và giá trị nó chứa, cũng cho ta những thông tin tốt.

### 3.2.3 Ước lượng điểm

Công việc mô tả thứ 3 là tạo ra ước lượng điểm của một thứ gì đó. Với toàn bộ phân phối posterior, bạn sẽ báo cáo giá trị nào? Có thể đây là một câu hỏi ngu ngốc, những nó rất khó trả lời. Ước lượng parameter kiểu Bayes cho kết quả là toàn bộ posterior, chứ không phải một con số, mà là một hàm số cho phép biến đổi một giá trị của parameter thành xác suất. Cho nên điều quan trọng ở đây là bạn không cần phải đưa ra ước lượng điểm. Nó không cần thiết và có thể gây người đọc hiểu sai vì mất thông tin.

Nhưng nếu bạn cần tạo ra một điểm ước lượng để mô tả posterior, bạn sẽ phải hỏi và trả lời nhiều câu hỏi hơn. Với ví dụ tung quả cầu với 3W trong 3 lần tung, bạn hãy xem 3 cách trả lời như sau. Đầu tiên là parameter có xác suất cao nhất, hay *maximum a posteriori (MAP)*.

```python
p_grid[np.argmax(posterior)]
# DeviceArray(1., dtype=float32)
```

Bạn cũng có thể tìm MAP trên mẫu:

```python
samples[np.argmax(gaussian_kde(samples, bw_method=0.01)(samples))]
# DeviceArray(0.988989, dtype=float32)
```

Nhưng tại sao lại điểm này, hay còn gọi là mode, tại sao không phải mean hay median?

```python
print(np.mean(samples))
print(np.median(samples))
# 0.8011085
# 0.8428428
```

![](/assets/images/figure 3-4.png)

Chúng là những ước lượng điểm. Nhưng cả 3 mode (MAP), mean, median khác nhau trong trường hợp này. Chọn bằng cách nào đây?

Một phương pháp có thể đi xa hơn cách chọn toàn bộ posterior là chọn một **HÀM LOSS**. Hàm loss là một quy tắc liên quan hao phí của một giá trị của parameter. Từ lâu các nhà thống kê đã hứng thú với hàm loss, và được thống kê Bayes cũng hỗ trợ, nhưng ít có ai dùng nó. Chủ yếu là do mỗi hàm loss khác nhau cho ước lượng điểm khác nhau.

Ví dụ có một trò chơi, hãy nói cho tôi biết giá trị *p* (tỉ lệ bề mặt nước ở quả cầu) nào, mà bạn nghĩ là đúng. Tôi sẽ trả bạn $100, nếu bạn đúng chính xác. Nhưng tôi sẽ trừ số tiền bạn được hưởng, tỉ lệ với khoảng cách từ lựa chọn của bạn đến giá trị đúng. Hay nói loss của bạn là giá trị tuyệt đối của *d - p*, *d8 là lựa chọn của bạn và *p* là câu trả lời đúng. Bạn có thể thay đổi số tiền, nhưng bạn cần để ý là loss tỉ lệ thuận với khoảng cách giữa *d* và *p*.

Bạn đã có posterior trong tay, làm sao để bạn tối ưu hoá tiền thắng. Thật ra, giá trị mà bạn sẽ tối ưu hoá tiền thắng (tối thiểu hoá mất mát) là nằm ở median của posterior.

Tính toán loss bằng cách lấy tổng của loss nhân với posterior tại cùng giá trị parameter.

```python
np.sum(posterior * np.abs(0.5 - p_grid))
# DeviceArray(0.3128752, dtype=float32)
```

Công thức trên tính trung bình của loss, với mỗi loss nhân với trọng số là xác suất posterior. Ta có thể ra dãy số loss với từng giá trị dự đoán.

```python
loss = vmap(lambda d: np.sum(posterior * np.abs(d - p_grid)))(p_grid)
```

Giá trị dự đoán mà có loss thấp nhất là:

```python
p_grid[np.argmin(loss)]
# DeviceArray(0.8408408, dtype=float32)
```

Đây thực ra là median của posterior, giá trị parameter mà chi xác suất posterior thành 2 phần có mật độ bằng nhau.

Vậy ta học được gì từ bài này? Để chọn một ước lượng điểm, một giá trị có thể mô tả được phân phối posterior, ta nên chọn một hàm loss. Hàm loss khác nhau thì cho ra ước lượng điểm khác nhau. Hai hàm loss phổ biến là giá trị tuyệt đối như trên, cho ra median, và quadratic loss $(d-p)^2$, cho ra mean của mẫu. Khi mà posterior đối xứng và nhìn giống normal, thì median và mean trùng nhau, giúp ta thoải mái hơn vì không cần hàm loss. Trong bài toán gốc tung quả cẩu (6W 3L), mean và median khác nhau ít.

Nguyên tắc, chi tiết ứng dụng trong ngữ cảnh có thể cần hàm loss đặc biệt, như trong quyết định cần giải toả hay không, dựa vào sức gió bão. Đồ vật và sinh mạng sẽ thiệt hại rất nhanh khi tốc độ gió tăng. Loss vẫn có khi ra lệnh giải toả, nhưng ít hơn. Cho nên hàm loss này khá bất xứng, tăng mạnh khi tốc độ gió vượt dự đoán, mà tăng chậm khi tốc độ gió giảm xuống hơn dự đoán. Trong ví dụ này, ước lượng điểm cần lớn hơn mean và median. Hơn nữa, vấn đề thực thiết là có nên giải toả hay không? Tạo ra ước lượng điểm cho tốc độ gió có thể không cần thiết.

Thông thường nhà nghiên cứu không quan tâm đến hàm loss. Nếu như có mean hoặc MAP trong báo cáo, có thể họ chẳng dành cho mục đích cụ thể nào cả, hoặc chỉ đơn thuần mô tả hình dạng của posterior. Bạn có thể nói rằng quyết định ở đây là có từ chối hay nhận giả thuyết không. Nhưng thách thức là nói về hao phí và lợi ít sẽ như thế nào, khi chấp nhận hay từ chối giả thuyết. Thông thường ta nên trình bày toàn bộ posterior và bản thân model và data, để người khác có thể xây dựng trên tác phẩm của mình. Một quyết định cho giả thuyết có thể ảnh hưởng tới sự sống.

Cần phải nhớ rằng, suy luận thống kê cho ta các câu hỏi mà chỉ có thể trả lời trong ngữ cảnh và mục đích ứng dụng cụ thể.

## <center>3.3 Lấy mẫu để mô phỏng dự đoán</center><a name="3.3"></a>

Một công việc thường gặp với mẫu là mô phỏng quan sát mới của model. Có 4 lý do để làm việc này:

1. *Thiết kế model.* Chúng ta có thể lấy mẫu không những từ posterior, mà còn từ prior. Nhìn model hoạt động, trước và sau khi có data, là một cách để hiểu rõ hơn tác động của prior. Khi có nhiều parameter hơn, thì quan hệ giữa chúng không rõ ràng lắm.
2. *Kiểm tra model.* Sau khi cập nhật data cho model, ta nên mô phỏng để kiểm tra là model fit đúng chưa và hành vi của model.
3. *Kiểm tra phần mềm.* Để đảm bảo phần mềm hoạt động tốt, thì việc lấy mẫu mô phỏng so sánh với model biết sẵn. 
4. *Thiết kế nghiên cứu.* Nếu bạn có thể mô phỏng mẫu từ giả thuyết, thì bạn có thể lượng giá hiệu quả của thiết kế nghiên cứu. Hay nói đơn giản là đây là *power analysis*, nhưng với tầm nhìn rộng hơn.
5. *Dự báo.* Ước lượng có thể dùng để dự đoán, cho trường hợp mới và quan sát trong tương lai. Những dự báo này có thể dùng trong thực dụng, cũng như đánh giá và nâng cấp.

### 3.3.1 Dummy data (data giả tạo)

Hãy tổng kết lại model tung quả cầu. Quả cầu có tỉ lệ bề mặt nước *p*, là parameter đích mà ta đang suy luận. Số quan sát ra "nước" và "đất" tỉ lệ thuận lần lượt với *p* và 1 - *p*.

Chú ý rằng những giả định nayỳ không chỉ cho chúng ta suy luận xác suất cho từng giá trị cụ thể của *p* sau khi có quan sát. Nó còn cho phép ta mô phỏng quan sát mới từ model. Nó cho phép vì hàm likelihood là hai chiều. Với một quan sát thực, hàm likelihood cho ta khả năng xuất hiện của quan sát đó. Với một parameter, hàm likelihood cho ta phân phối của mọi quan sát mà ta có thể lấy từ phân phối, để tạo quan sát mới. Cho nên, model Bayes luôn có tính *tạo mới (generative)*, có thể mô phỏng dự đoán. Rất nhiều model non-Bayes có tính tạo mới, và cũng nhiều model không có tính này.

Ta gọi data được mô phỏng này là **DUMMY DATA**. Với ví dụ tung quả cầu, dummy data từ likelihood binomial:

$$ Pr(W|N,p) = \frac{N!}{W!(N-W)!} p^W (1-p)^{N-W} $$

Giả sử N=2, 2 lần tung quả cầu, có 3 khả năng xảy ra: 0W, 1W, 2W. Bạn có thể tính xác suất của từng khả năng, với giá trị *p* bất kỳ. Ví dụ với *p*=0.7:

```python
np.exp(dist.Binomial(total_count=2, probs=0.7).log_prob(np.arange(3)))
# DeviceArray([0.08999996, 0.42000008, 0.48999974], dtype=float32)
```

Có nghĩa là có 9% cơ hội để có quan sát w=0, 42% với w=1 và 49% với w=2. Khi thay đổi *p* thì ta có kết quả khác.

Giờ ta sẽ lấy mẫu thử, bằng các xác suất này.

```python
dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(0))
# DeviceArray(1., dtype=float32)
```

Có nghĩa là 1W trong 2 lần tung quả cầu. Ta có thể sample nhiều mẫu hơn:

```python
dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(2), (10,))
# DeviceArray([0., 2., 2., 2., 1., 2., 2., 1., 0., 0.], dtype=float32)
```

Bây giờ tạo 100,000 dummy data, và xem tần suất của nó:

```python
dummy_w = dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(0), (100000,))
onp.unique(dummy_w, return_counts=True)[1] / 1e5
# array([0.0883 , 0.42101, 0.49069])
```

Những con số này gần giống với kết quả phân tích trực tiếp từ likelihood. Giờ ta sẽ xem kết quả từ 9 lần tung.

```python
dummy_w = dist.Binomial(total_count=9, probs=0.7).sample(PRNGKey(0), (100000,))
ax = az.plot_dist(dummy_w.copy(), kind='hist', hist_kwargs={"rwidth": 0.1})
ax.set_xlabel("dummy water count", fontsize=14);
```

<svg height="266.43875pt" version="1.1" viewBox="0 0 383.871875 266.43875" width="383.871875pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style></defs><g id="figure_1"><g id="patch_1"><path d="M 0 266.43875 
L 383.871875 266.43875 
L 383.871875 0 
L 0 0 
z
" style="fill:none;"></path></g><g id="axes_1"><g id="patch_2"><path d="M 41.871875 224.64 
L 376.671875 224.64 
L 376.671875 7.2 
L 41.871875 7.2 
z
" style="fill:#eeeeee;"></path></g><g id="matplotlib.axis_1"><g id="xtick_1"><g id="line2d_1"><path clip-path="url(#p553551725c)" d="M 58.762384 224.64 
L 58.762384 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_2"></g><g id="text_1"><!-- 0 --><defs><path d="M 31.78125 66.40625 
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
" id="DejaVuSans-48"></path></defs><g style="fill:#262626;" transform="translate(54.308634 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use></g></g></g><g id="xtick_2"><g id="line2d_3"><path clip-path="url(#p553551725c)" d="M 92.208938 224.64 
L 92.208938 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_4"></g><g id="text_2"><!-- 1 --><defs><path d="M 12.40625 8.296875 
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
" id="DejaVuSans-49"></path></defs><g style="fill:#262626;" transform="translate(87.755188 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-49"></use></g></g></g><g id="xtick_3"><g id="line2d_5"><path clip-path="url(#p553551725c)" d="M 125.655491 224.64 
L 125.655491 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_6"></g><g id="text_3"><!-- 2 --><defs><path d="M 19.1875 8.296875 
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
" id="DejaVuSans-50"></path></defs><g style="fill:#262626;" transform="translate(121.201741 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-50"></use></g></g></g><g id="xtick_4"><g id="line2d_7"><path clip-path="url(#p553551725c)" d="M 159.102045 224.64 
L 159.102045 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_8"></g><g id="text_4"><!-- 3 --><defs><path d="M 40.578125 39.3125 
Q 47.65625 37.796875 51.625 33 
Q 55.609375 28.21875 55.609375 21.1875 
Q 55.609375 10.40625 48.1875 4.484375 
Q 40.765625 -1.421875 27.09375 -1.421875 
Q 22.515625 -1.421875 17.65625 -0.515625 
Q 12.796875 0.390625 7.625 2.203125 
L 7.625 11.71875 
Q 11.71875 9.328125 16.59375 8.109375 
Q 21.484375 6.890625 26.8125 6.890625 
Q 36.078125 6.890625 40.9375 10.546875 
Q 45.796875 14.203125 45.796875 21.1875 
Q 45.796875 27.640625 41.28125 31.265625 
Q 36.765625 34.90625 28.71875 34.90625 
L 20.21875 34.90625 
L 20.21875 43.015625 
L 29.109375 43.015625 
Q 36.375 43.015625 40.234375 45.921875 
Q 44.09375 48.828125 44.09375 54.296875 
Q 44.09375 59.90625 40.109375 62.90625 
Q 36.140625 65.921875 28.71875 65.921875 
Q 24.65625 65.921875 20.015625 65.03125 
Q 15.375 64.15625 9.8125 62.3125 
L 9.8125 71.09375 
Q 15.4375 72.65625 20.34375 73.4375 
Q 25.25 74.21875 29.59375 74.21875 
Q 40.828125 74.21875 47.359375 69.109375 
Q 53.90625 64.015625 53.90625 55.328125 
Q 53.90625 49.265625 50.4375 45.09375 
Q 46.96875 40.921875 40.578125 39.3125 
z
" id="DejaVuSans-51"></path></defs><g style="fill:#262626;" transform="translate(154.648295 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-51"></use></g></g></g><g id="xtick_5"><g id="line2d_9"><path clip-path="url(#p553551725c)" d="M 192.548598 224.64 
L 192.548598 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_10"></g><g id="text_5"><!-- 4 --><defs><path d="M 37.796875 64.3125 
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
" id="DejaVuSans-52"></path></defs><g style="fill:#262626;" transform="translate(188.094848 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-52"></use></g></g></g><g id="xtick_6"><g id="line2d_11"><path clip-path="url(#p553551725c)" d="M 225.995152 224.64 
L 225.995152 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_12"></g><g id="text_6"><!-- 5 --><defs><path d="M 10.796875 72.90625 
L 49.515625 72.90625 
L 49.515625 64.59375 
L 19.828125 64.59375 
L 19.828125 46.734375 
Q 21.96875 47.46875 24.109375 47.828125 
Q 26.265625 48.1875 28.421875 48.1875 
Q 40.625 48.1875 47.75 41.5 
Q 54.890625 34.8125 54.890625 23.390625 
Q 54.890625 11.625 47.5625 5.09375 
Q 40.234375 -1.421875 26.90625 -1.421875 
Q 22.3125 -1.421875 17.546875 -0.640625 
Q 12.796875 0.140625 7.71875 1.703125 
L 7.71875 11.625 
Q 12.109375 9.234375 16.796875 8.0625 
Q 21.484375 6.890625 26.703125 6.890625 
Q 35.15625 6.890625 40.078125 11.328125 
Q 45.015625 15.765625 45.015625 23.390625 
Q 45.015625 31 40.078125 35.4375 
Q 35.15625 39.890625 26.703125 39.890625 
Q 22.75 39.890625 18.8125 39.015625 
Q 14.890625 38.140625 10.796875 36.28125 
z
" id="DejaVuSans-53"></path></defs><g style="fill:#262626;" transform="translate(221.541402 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-53"></use></g></g></g><g id="xtick_7"><g id="line2d_13"><path clip-path="url(#p553551725c)" d="M 259.441705 224.64 
L 259.441705 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_14"></g><g id="text_7"><!-- 6 --><defs><path d="M 33.015625 40.375 
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
" id="DejaVuSans-54"></path></defs><g style="fill:#262626;" transform="translate(254.987955 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-54"></use></g></g></g><g id="xtick_8"><g id="line2d_15"><path clip-path="url(#p553551725c)" d="M 292.888259 224.64 
L 292.888259 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_16"></g><g id="text_8"><!-- 7 --><defs><path d="M 8.203125 72.90625 
L 55.078125 72.90625 
L 55.078125 68.703125 
L 28.609375 0 
L 18.3125 0 
L 43.21875 64.59375 
L 8.203125 64.59375 
z
" id="DejaVuSans-55"></path></defs><g style="fill:#262626;" transform="translate(288.434509 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-55"></use></g></g></g><g id="xtick_9"><g id="line2d_17"><path clip-path="url(#p553551725c)" d="M 326.334812 224.64 
L 326.334812 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_18"></g><g id="text_9"><!-- 8 --><defs><path d="M 31.78125 34.625 
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
" id="DejaVuSans-56"></path></defs><g style="fill:#262626;" transform="translate(321.881062 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-56"></use></g></g></g><g id="xtick_10"><g id="line2d_19"><path clip-path="url(#p553551725c)" d="M 359.781366 224.64 
L 359.781366 7.2 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_20"></g><g id="text_10"><!-- 9 --><defs><path d="M 10.984375 1.515625 
L 10.984375 10.5 
Q 14.703125 8.734375 18.5 7.8125 
Q 22.3125 6.890625 25.984375 6.890625 
Q 35.75 6.890625 40.890625 13.453125 
Q 46.046875 20.015625 46.78125 33.40625 
Q 43.953125 29.203125 39.59375 26.953125 
Q 35.25 24.703125 29.984375 24.703125 
Q 19.046875 24.703125 12.671875 31.3125 
Q 6.296875 37.9375 6.296875 49.421875 
Q 6.296875 60.640625 12.9375 67.421875 
Q 19.578125 74.21875 30.609375 74.21875 
Q 43.265625 74.21875 49.921875 64.515625 
Q 56.59375 54.828125 56.59375 36.375 
Q 56.59375 19.140625 48.40625 8.859375 
Q 40.234375 -1.421875 26.421875 -1.421875 
Q 22.703125 -1.421875 18.890625 -0.6875 
Q 15.09375 0.046875 10.984375 1.515625 
z
M 30.609375 32.421875 
Q 37.25 32.421875 41.125 36.953125 
Q 45.015625 41.5 45.015625 49.421875 
Q 45.015625 57.28125 41.125 61.84375 
Q 37.25 66.40625 30.609375 66.40625 
Q 23.96875 66.40625 20.09375 61.84375 
Q 16.21875 57.28125 16.21875 49.421875 
Q 16.21875 41.5 20.09375 36.953125 
Q 23.96875 32.421875 30.609375 32.421875 
z
" id="DejaVuSans-57"></path></defs><g style="fill:#262626;" transform="translate(355.327616 238.777813)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-57"></use></g></g></g><g id="text_11"><!-- dummy water count --><defs><path d="M 45.40625 46.390625 
L 45.40625 75.984375 
L 54.390625 75.984375 
L 54.390625 0 
L 45.40625 0 
L 45.40625 8.203125 
Q 42.578125 3.328125 38.25 0.953125 
Q 33.9375 -1.421875 27.875 -1.421875 
Q 17.96875 -1.421875 11.734375 6.484375 
Q 5.515625 14.40625 5.515625 27.296875 
Q 5.515625 40.1875 11.734375 48.09375 
Q 17.96875 56 27.875 56 
Q 33.9375 56 38.25 53.625 
Q 42.578125 51.265625 45.40625 46.390625 
z
M 14.796875 27.296875 
Q 14.796875 17.390625 18.875 11.75 
Q 22.953125 6.109375 30.078125 6.109375 
Q 37.203125 6.109375 41.296875 11.75 
Q 45.40625 17.390625 45.40625 27.296875 
Q 45.40625 37.203125 41.296875 42.84375 
Q 37.203125 48.484375 30.078125 48.484375 
Q 22.953125 48.484375 18.875 42.84375 
Q 14.796875 37.203125 14.796875 27.296875 
z
" id="DejaVuSans-100"></path><path d="M 8.5 21.578125 
L 8.5 54.6875 
L 17.484375 54.6875 
L 17.484375 21.921875 
Q 17.484375 14.15625 20.5 10.265625 
Q 23.53125 6.390625 29.59375 6.390625 
Q 36.859375 6.390625 41.078125 11.03125 
Q 45.3125 15.671875 45.3125 23.6875 
L 45.3125 54.6875 
L 54.296875 54.6875 
L 54.296875 0 
L 45.3125 0 
L 45.3125 8.40625 
Q 42.046875 3.421875 37.71875 1 
Q 33.40625 -1.421875 27.6875 -1.421875 
Q 18.265625 -1.421875 13.375 4.4375 
Q 8.5 10.296875 8.5 21.578125 
z
M 31.109375 56 
z
" id="DejaVuSans-117"></path><path d="M 52 44.1875 
Q 55.375 50.25 60.0625 53.125 
Q 64.75 56 71.09375 56 
Q 79.640625 56 84.28125 50.015625 
Q 88.921875 44.046875 88.921875 33.015625 
L 88.921875 0 
L 79.890625 0 
L 79.890625 32.71875 
Q 79.890625 40.578125 77.09375 44.375 
Q 74.3125 48.1875 68.609375 48.1875 
Q 61.625 48.1875 57.5625 43.546875 
Q 53.515625 38.921875 53.515625 30.90625 
L 53.515625 0 
L 44.484375 0 
L 44.484375 32.71875 
Q 44.484375 40.625 41.703125 44.40625 
Q 38.921875 48.1875 33.109375 48.1875 
Q 26.21875 48.1875 22.15625 43.53125 
Q 18.109375 38.875 18.109375 30.90625 
L 18.109375 0 
L 9.078125 0 
L 9.078125 54.6875 
L 18.109375 54.6875 
L 18.109375 46.1875 
Q 21.1875 51.21875 25.484375 53.609375 
Q 29.78125 56 35.6875 56 
Q 41.65625 56 45.828125 52.96875 
Q 50 49.953125 52 44.1875 
z
" id="DejaVuSans-109"></path><path d="M 32.171875 -5.078125 
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
" id="DejaVuSans-121"></path><path id="DejaVuSans-32"></path><path d="M 4.203125 54.6875 
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
" id="DejaVuSans-119"></path><path d="M 34.28125 27.484375 
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
" id="DejaVuSans-97"></path><path d="M 18.3125 70.21875 
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
" id="DejaVuSans-116"></path><path d="M 56.203125 29.59375 
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
" id="DejaVuSans-101"></path><path d="M 41.109375 46.296875 
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
" id="DejaVuSans-114"></path><path d="M 48.78125 52.59375 
L 48.78125 44.1875 
Q 44.96875 46.296875 41.140625 47.34375 
Q 37.3125 48.390625 33.40625 48.390625 
Q 24.65625 48.390625 19.8125 42.84375 
Q 14.984375 37.3125 14.984375 27.296875 
Q 14.984375 17.28125 19.8125 11.734375 
Q 24.65625 6.203125 33.40625 6.203125 
Q 37.3125 6.203125 41.140625 7.25 
Q 44.96875 8.296875 48.78125 10.40625 
L 48.78125 2.09375 
Q 45.015625 0.34375 40.984375 -0.53125 
Q 36.96875 -1.421875 32.421875 -1.421875 
Q 20.0625 -1.421875 12.78125 6.34375 
Q 5.515625 14.109375 5.515625 27.296875 
Q 5.515625 40.671875 12.859375 48.328125 
Q 20.21875 56 33.015625 56 
Q 37.15625 56 41.109375 55.140625 
Q 45.0625 54.296875 48.78125 52.59375 
z
" id="DejaVuSans-99"></path><path d="M 30.609375 48.390625 
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
" id="DejaVuSans-111"></path><path d="M 54.890625 33.015625 
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
" id="DejaVuSans-110"></path></defs><g style="fill:#262626;" transform="translate(138.470156 256.327188)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-100"></use><use x="63.476562" xlink:href="#DejaVuSans-117"></use><use x="126.855469" xlink:href="#DejaVuSans-109"></use><use x="224.267578" xlink:href="#DejaVuSans-109"></use><use x="321.679688" xlink:href="#DejaVuSans-121"></use><use x="380.859375" xlink:href="#DejaVuSans-32"></use><use x="412.646484" xlink:href="#DejaVuSans-119"></use><use x="494.433594" xlink:href="#DejaVuSans-97"></use><use x="555.712891" xlink:href="#DejaVuSans-116"></use><use x="594.921875" xlink:href="#DejaVuSans-101"></use><use x="656.445312" xlink:href="#DejaVuSans-114"></use><use x="697.558594" xlink:href="#DejaVuSans-32"></use><use x="729.345703" xlink:href="#DejaVuSans-99"></use><use x="784.326172" xlink:href="#DejaVuSans-111"></use><use x="845.507812" xlink:href="#DejaVuSans-117"></use><use x="908.886719" xlink:href="#DejaVuSans-110"></use><use x="972.265625" xlink:href="#DejaVuSans-116"></use></g></g></g><g id="matplotlib.axis_2"><g id="ytick_1"><g id="line2d_21"><path clip-path="url(#p553551725c)" d="M 41.871875 224.64 
L 376.671875 224.64 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_22"></g><g id="text_12"><!-- 0.00 --><defs><path d="M 10.6875 12.40625 
L 21 12.40625 
L 21 0 
L 10.6875 0 
z
" id="DejaVuSans-46"></path></defs><g style="fill:#262626;" transform="translate(7.2 229.958906)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="ytick_2"><g id="line2d_23"><path clip-path="url(#p553551725c)" d="M 41.871875 185.972864 
L 376.671875 185.972864 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_24"></g><g id="text_13"><!-- 0.05 --><g style="fill:#262626;" transform="translate(7.2 191.29177)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-48"></use><use x="159.033203" xlink:href="#DejaVuSans-53"></use></g></g></g><g id="ytick_3"><g id="line2d_25"><path clip-path="url(#p553551725c)" d="M 41.871875 147.305728 
L 376.671875 147.305728 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_26"></g><g id="text_14"><!-- 0.10 --><g style="fill:#262626;" transform="translate(7.2 152.624634)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-49"></use><use x="159.033203" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="ytick_4"><g id="line2d_27"><path clip-path="url(#p553551725c)" d="M 41.871875 108.638592 
L 376.671875 108.638592 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_28"></g><g id="text_15"><!-- 0.15 --><g style="fill:#262626;" transform="translate(7.2 113.957498)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-49"></use><use x="159.033203" xlink:href="#DejaVuSans-53"></use></g></g></g><g id="ytick_5"><g id="line2d_29"><path clip-path="url(#p553551725c)" d="M 41.871875 69.971455 
L 376.671875 69.971455 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_30"></g><g id="text_16"><!-- 0.20 --><g style="fill:#262626;" transform="translate(7.2 75.290362)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use><use x="159.033203" xlink:href="#DejaVuSans-48"></use></g></g></g><g id="ytick_6"><g id="line2d_31"><path clip-path="url(#p553551725c)" d="M 41.871875 31.304319 
L 376.671875 31.304319 
" style="fill:none;stroke:#ffffff;stroke-linecap:round;stroke-width:0.8;"></path></g><g id="line2d_32"></g><g id="text_17"><!-- 0.25 --><g style="fill:#262626;" transform="translate(7.2 36.623226)scale(0.14 -0.14)"><use xlink:href="#DejaVuSans-48"></use><use x="63.623047" xlink:href="#DejaVuSans-46"></use><use x="95.410156" xlink:href="#DejaVuSans-50"></use><use x="159.033203" xlink:href="#DejaVuSans-53"></use></g></g></g></g><g id="patch_3"><path clip-path="url(#p553551725c)" d="M 57.090057 224.64 
L 60.434712 224.64 
L 60.434712 224.6168 
L 57.090057 224.6168 
z
" style="fill:#2a2eec;"></path></g><g id="patch_4"><path clip-path="url(#p553551725c)" d="M 90.53661 224.64 
L 93.881266 224.64 
L 93.881266 224.407997 
L 90.53661 224.407997 
z
" style="fill:#2a2eec;"></path></g><g id="patch_5"><path clip-path="url(#p553551725c)" d="M 123.983164 224.64 
L 127.327819 224.64 
L 127.327819 221.747698 
L 123.983164 221.747698 
z
" style="fill:#2a2eec;"></path></g><g id="patch_6"><path clip-path="url(#p553551725c)" d="M 157.429717 224.64 
L 160.774373 224.64 
L 160.774373 207.989931 
L 157.429717 207.989931 
z
" style="fill:#2a2eec;"></path></g><g id="patch_7"><path clip-path="url(#p553551725c)" d="M 190.876271 224.64 
L 194.220926 224.64 
L 194.220926 167.791576 
L 190.876271 167.791576 
z
" style="fill:#2a2eec;"></path></g><g id="patch_8"><path clip-path="url(#p553551725c)" d="M 224.322824 224.64 
L 227.667479 224.64 
L 227.667479 92.792799 
L 224.322824 92.792799 
z
" style="fill:#2a2eec;"></path></g><g id="patch_9"><path clip-path="url(#p553551725c)" d="M 257.769377 224.64 
L 261.114033 224.64 
L 261.114033 17.554286 
L 257.769377 17.554286 
z
" style="fill:#2a2eec;"></path></g><g id="patch_10"><path clip-path="url(#p553551725c)" d="M 291.215931 224.64 
L 294.560586 224.64 
L 294.560586 17.786289 
L 291.215931 17.786289 
z
" style="fill:#2a2eec;"></path></g><g id="patch_11"><path clip-path="url(#p553551725c)" d="M 324.662484 224.64 
L 328.00714 224.64 
L 328.00714 104.671343 
L 324.662484 104.671343 
z
" style="fill:#2a2eec;"></path></g><g id="patch_12"><path clip-path="url(#p553551725c)" d="M 358.109038 224.64 
L 361.453693 224.64 
L 361.453693 193.698558 
L 358.109038 193.698558 
z
" style="fill:#2a2eec;"></path></g><g id="patch_13"><path d="M 41.871875 224.64 
L 41.871875 7.2 
" style="fill:none;"></path></g><g id="patch_14"><path d="M 376.671875 224.64 
L 376.671875 7.2 
" style="fill:none;"></path></g><g id="patch_15"><path d="M 41.871875 224.64 
L 376.671875 224.64 
" style="fill:none;"></path></g><g id="patch_16"><path d="M 41.871875 7.2 
L 376.671875 7.2 
" style="fill:none;"></path></g></g></g><defs><clipPath id="p553551725c"><rect height="217.44" width="334.8" x="41.871875" y="7.2"></rect></clipPath></defs></svg>

Chú ý rằng là quan sát mong đợi phần lớn thời gian điều không có tỉ lệ nước chính xác = 0.7. Đó là đặc tính của quan sát: có mối quan hệ one-to-many giữa data và quy trình tạo data. Bạn nên thí nghiệm với số lượng mẫu khác nhau, cũng như giá trị parameter khác nhau để xem sự thay đổi hình dáng và vị trí của phân phối của mẫu.

>**Nghĩ lại: Phân phối mẫu (Sampling distributión).** Phân phối mẫu là nền tảng của nhiều thống kê non-Bayes. Trong thống kê ấy, suy luận parameter dựa trên phân phối này. Trong sách này, suy luận parameter không được làm trực tiếp trên phân phối mẫu. Ta không lấy mẫu từ posterior, mà tạo posterior một cách logic. Và lấy mẫu từ posterior để ủng hộ suy luận.

### 3.3.2 Kiểm tra model

Kiểm tra model là:

1. Đảm bảo model hoạt động đúng
2. Lượng giá sự thích hợp của model với mục đích cụ thể.

#### 3.3.2.1 Phần mềm có hoạt động đúng?

Trong trường hợp đơn giản nhất, ta chỉ kiểm tra phần mềm hoạt động đúng hay chưa bằng so sánh sự tương ứng giữa dự đoán và data để fit vào model. Không cần phải chính xác về mặt toán học. Nếu không có sự tương ứng nào cả, thì có lẽ phần mềm đã có lỗi.

Thật ra không có cách nào để đảm bảo phần mềm hoạt động đúng. Ngay cả phương pháp trên cũng có thể có lỗi nào đó. Khi bạn làm việc với model đa tầng, bạn sẽ thấy rõ điều này. Nhưng với một bước nhỏ đơn giản này, bạn có thể phát hiện những lỗi sai ngớ ngẩn mà loài người hay bị.

#### 3.3.2.2 Model có thích hợp chưa?

Sau khi có được phân phối posterior đúng, bởi vì phần mềm hoạt động đúng, ta nên xem xét các khía cạnh của data mà model không giải thích được. Mục đích không phải là kiểm tra giả định model là đúng, bởi vì tất cả đều là sai. Điều cần làm là tại sao model lại thất bại trong việc mô tả data, để hướng dẫn hoàn thiện và nâng cấp model.

Mọi model đều có những sai sót ở khía cạnh nào đó, nên bạn phải biết phán đoán để nhìn nhận lỗi sai có quan trọng hay không. Ít nhà nghiên cứu muốn tạo model chỉ để tái hiện lại mẫu thu thập được. Cho nên dự đoán không hoàn hảo không phải điều xấu. Cụ thể ta muốn dự đoán được một quan sát tương lai và hiểu rõ hơn để ta có thể tuỳ chỉnh thể giới thực. 

Bây giờ, ta sẽ học cách kết hợp cách lấy mẫu quan sát giả tạo, với lấy mẫu các parameter từ posterior. Ta nghĩ sẽ tốt hơn khi dùng toàn bộ phân phối posterior hơn là một ước lượng điểm nào đó. Tại sao? Bởi vì có một lượng lớn thông tin về tính bất định trong toàn bộ phân phối posterior. Ta sẽ mất thông tin ấy khi chỉ dùng một điểm nào đó và tính toán dựa trên nó. Chính sự mất mát thông tin này làm ta trở nên tự tin thái quá.

Ví dụ như trong model tung quả cầu. Quan sát ở đây là số đếm của "nước", trên số lần tung quả cầu. Dự đoán của model dựa trên tính bất định của 2 yếu tố.

Trước tiên là yếu tố bất định từ quan sát. Với mỗi giá trị của *p*, có một kiểu tần suất các quan sát mà model mong đợi, giống như model khu vườn phân nhánh. Nó cũng là lấy mẫu từ likelihood mà bạn vừa mới thấy. Đây là yếu tố bất định trong dự báo quan sát mới, bởi vì khi bạn đã xác định *p*, bạn không biết lần tung sau như thế nào (nếu *p* không phải 0 hay 1).

Thứ hai là yếu tố bất định từ *p* được biểu diễn bằng posterior. Chính vì sự bất định *p* này, cho nên có sự bất định lên mọi thứ liên quan đến *p*. Và sự bất định này sẽ tương tác với sự bất định từ quan sát, khi ta cần tìm hiểu model nói gì cho chúng ta về kết quả của nó.

Ta cần phải truyền tải tính bất định từ parameter đến dự đoán, bằng cách lấy trung bình tất cả mật độ posterior cho *p*, khi tính ra dự đoán. Với mỗi giá trị của *p*, có một phân phối kết cục riêng. Và nếu bạn có thể tính được phân phối mẫu của kết cục tại mỗi điểm *p*, bạn có thể lấy trung bình tất cả phân phối dự đoán, bằng xác suất posterior, gọi là phân phối dự đoán posterior **(POSTERIOR PREDICTIVE DISTRIBUTION)**.

![](/assets/images/figure 3-6.png)

Hình trên mô tả quy trình đã nêu. Trên cùng là posterior với 10 giá trị của parameter. Tại mỗi giá trị thì có một phân phối mẫu dự đoán riêng, được vẽ ở hàng giữa. Quan sát luôn có tính bất định cho một giá trị *p* bất kỳ, nhưng sẽ thay đổi hình dạng tuỳ theo nó. Hàng cuối, là trung bình có trọng số của toàn bộ phân phối mẫu, sau khi dùng xác suất parameter từ posterior.

Kết quả cuối cùng là phân phối dự đoán posterior. Nó rất thành thật. Mặc dù model đã làm rất tốt để dự đoán data, nhưng nó vẫn rất loãng. Nếu bạn chỉ dùng một giá trị để tính dự đoán, ví dụ như dùng giá trị nằm ở đỉnh posterior, bạn sẽ bị tin cậy thái quá vào phân phối của dự đoán, vì nó hẹp so với posterior predictive distribution. Hậu quả của sự tin cậy thái quá cho bạn tầm nhìn sai vào model phù hợp vào data hơn vốn dĩ của nó. Sai lầm trực giác này đã bỏ đi tính bất định của parameter.

Vậy cụ thể tính toán như thế nào?

```python
w = dist.Binomial(total_count=9, probs=samples).sample(PRNGKey(0))
```

`samples` ở trên là kết quả lấy mẫu *p* từ posterior đã được tạo ở phần trước. Với mỗi giá trị trong `samples`, một quan sát từ binomial được tạo. Bởi vì `samples` đã tuân theo phân phối posterior, nên kết quả mô phỏng `w` đã trung bình hoá posterior. Bạn có thể mô tả kết quả này giống như từng làm với phân phối posterior, như ước lượng khoảng và ước lượng điểm.

Kết quả từ mô phỏng dự đoán khá phù hợp với data quan sát được trong trường hợp này - số đếm 6 nằm ngay ở chính giữa của phân phối mô phỏng. Mẫu mô phỏng này khá loãng, do là nó từ bản thân quy trình lấy mẫu binomial, chứ không phải từ tính bất định của *p*. Nhưng còn khá non để kết luận model này là hoàn hảo. Đến giờ, chúng ta chỉ nhìn data cũng giống như model nhìn data vậy: Mỗi lần tung đều không liên quan với nhau. Giả định này cần xem xét lại. Nếu người tung quả cầu này là người cẩn thận, người đó có thể gây ra sự tương quan và có cùng kiểu kết quả trong chuỗi tung quả cầu. Hãy suy nghĩ nếu bề mặt quả cầu có một nửa là Thái Bình Dương. Như vậy, nước và đất không còn phân phối uniform nữa, và nếu quả cầu không xoay đủ trong không trung, vị trí ban đầu khi tung có thể ảnh hưởng đến kết quả sau cùng.

Vậy với mục đích là tìm hiểu khía cạnh của dự đoán mà model có thể sai, hãy nhìn data với 2 cách. Nhớ lại trình tự của 9 lần tung là WLWWWLWLW. Trước tiên, hãy xem xét độ dài lớn nhất của nước hoặc đất. Nó cho ta sự đo đạc thô của tương quan giữa mỗi lần tung. Trong data, độ dài lớn nhất là 3W. Thứ hai, xem xét số lần thay đổi của nước và đất. Đây cũng là một phương pháp đo lường tương quan giữa các mẫu. Trong data thì số lần thay đổi là 6. Không có gì đặc biệt trong 2 cách trình bày data này.

![](/assets/images/figure 3-7.png)

Sau đó ta trình bày posterior predict distribution nhưng với 2 cách trình bày trên. Bên trái ta thấy độ dài lớn nhất của đât và nước của các mẫu đã lấy, và giá trị 3 được tô đậm. Lần nữa, quan sát trong data trùng với quan sát mô phỏng, nhưng nó khá loãng. Bên phải, số lần thay đổi giữa nước và đất được vẽ lên, và giá trị 6 được tô đậm. Bây giờ thì dự đoán không đồng bộ với data, vì mẫu mô phỏng quan sát được thường có ít số lần thay đổi hơn so với data. Điều này có nghĩa mỗi lần tung quả cầu có tương quan âm với nhau.

Có phải model bạn không tốt? Tuỳ. Model có thể sai theo một khía cạnh nào đó. Nhưng điều này có dẫn ta đến thay đổi model không thì còn tuỳ thuộc vào mục đích sử dụng. Trong trường hợp này, nếu việc tung quả cầu thường đổi từ W sang L hay L sang W, thì mỗi lần tung sẽ cho ít thông tin hơn tỉ lệ diện tích bề mặt. Trong thời gian dài, ngay cả model sai mà ta dùng vẫn cho kết quả tỉ lệ bề mặt đúng. Nhưng nó sẽ chậm hơn so với phân phối posterior.

>**Nghĩ lại: giá trị cực là gì?** Một phương pháp thông thường để đo đạc biến thiên của quan sát từ model là đếm tần suất của vùng đuôi chứa data và data hiếm gặp hơn. p-value là ví dụ của xác suất đuôi này.  
Có nhiều cách để nhìn data và định nghĩa "giá trị cực". p-value nhìn data theo kiểu data mong muốn, nên nó là một dạng kiểm tra model rất yếu. Ví dụ như hình posterior predictive distribution là cách tốt nhất để kiểm tra model. Cách định nghĩa khác "giá trị cực" có thể là một thử thách nghiêm trọng cho model. Ví dụ như trong hình độ dài lớn nhất và số lần thay đổi.  
Fit model là một quy trình có tính đối tượng - mọi người và golem có thể cập nhật Bayesian mà không phụ thuộc vào tuỳ thích cá nhân. Nhưng kiểm tra model là chủ quan, và chính yếu tố này là điểm mạnh của model Bayes, vì yếu tố chủ quan cần kinh nghiệm và kiến thưc của người sử dụng. Người sử dụng có thể tưởng tượng nhiều cách kiểm tra năng suất của model. Bởi vì golem không có trí tưởng tượng, ta cần sự tự do để lồng ghép sự tưởng tượng vào. Theo cách này, model Bayes có cả tính chủ quan và tính đối tượng.

## <center>3.4 Tổng kết</center><a name="3.4"></a>

Chương này giới thiệu những quy trình cơ bản lấy xử lý phân phối posterior. Nguyên lý chính là lấy mẫu các giá trị parameter rút ra từ phân phối posterior. Làm việc với mẫu sẽ chuyển cách làm việc với tích phân thành làm việc với phép cộng. Những mẫu này có thể dùng để tạo khoảng tin cậy, ước lượng điểm, dự đoán từ posterior, cũng như nhiều cách mô phỏng khác.

Kiểm tra dự đoán posterior kết hợp tính bất định của parameter từ posterior, với tính bất định của kết quả từ hàm likelihood có từ giả định. Việc kiểm tra rất có ích trong việc kiểm tra phần mềm hoạt động đúng cũng như sự phù hợp của model.

Khi model phức tạp hơn, mô phỏng dự đoán posterior sẽ dùng được cho nhiều ứng dụng khác. Ngay cả để hiểu model cần phải mô phỏng quan sát.