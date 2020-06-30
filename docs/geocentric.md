---
title: "Geocentric Models"
description: "Model Địa Tâm"
---

> Bài viết dịch bởi người không chuyên, độc giả nào có góp ý xin phản hồi lại.

```python
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import gaussian_kde

from jax import lax, vmap
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.contrib.autoguide import (AutoContinuousELBO,
                                       AutoLaplaceApproximation)
from numpyro.diagnostics import hpdi, print_summary
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, init_to_value
import numpyro.optim as optim
```
- [4.1 Tại sao phân phối normal lại normal](#1)
- [4.2 Ngôn ngữ mô tả model](#2)
- [4.3 Model Gaussian chiều cao](#3)
- [4.4 Linear Model](#4)
- [4.5 Cong từ thẳng](#5)

Ptolemy có một lịch sử khắc nghiệt. Claudius Ptolemy (90-168 sau công nguyên) là một nhà toán học và chiêm tinh gia người Ai Cập, ông được biết đến với model địa tâm trong hệ mặt trời. Ở hiện đại, nếu khoa học muốn chế giễu ai đó, thì họ sẽ ví ông như những kẻ tin vào thuyết địa tâm. Nhưng ông là một thiên tài. Model toán học về chuyển đạo hành tin cực kỳ chính xác. Để có được độ chính xác cao, ông dùng thiết bị tên là *epicycle*, tức vòng tròn trên vòng tròn. Có thể có epi-epicycle, vòng tròn trên vòng tròn trên vòng tròn. Nếu số lượng vòng tròn đủ và đúng, model của Ptolemy có thể dự báo chính xác chuyển đạo hành tinh với chính xác cao. Và nên model của ông đã được sử dụng hơn một nghìn năm. Ptolemy và những người như ông đã xây dựng những model này không cần sự hỗ trợ của máy tính. Ai ai cũng sẽ xấu hổ nếu so sánh với Ptolemy.

Câu hỏi quan trọng là model địa tâm là sai, ở nhiều phương diện. Nếu bạn dùng nó để vẽ chuyển đạo của Sao Hoả, bạn sẽ vẽ sai vị trí hành tinh đỏ này một khoảng rất dài. Nhưng model vẫn rất tốt để phát hiện Sao Hoả trên bầu trời. Mặc dù có thể model cần phải được tinh chỉnh lại mỗi thập kỷ, tuỳ vào vật thể nào mà bạn muốn định vị. Nhưng model địa tâm vẫn tiếp tục cho ra dự báo chính xác, với những câu hỏi trong giới hạn nhỏ.

![](/assets/images/figure 4-1.png)

Kỹ thuật epicycle có vẻ điên rồ, khi mà bạn biết chính xác cấu trúc của hệ mặt trời. Nhưng trong cổ đại, con người đã dùng nó như biện pháp ước lượng tổng quát hoá. Cho rằng có số lượng vòng tròn đủ trong một không gian đủ, cách làm của Ptolemy giống như *Fourier serries*, một cách phân tách một hàm tuần hoàn (như quỹ đạo) thành một tập hợp hàm sin và hàm cos. Cho nên các hành tinh thực có sắp xếp như thế nào thì model địa tâm vẫn có thể dùng để mô tả quỹ đạo của chúng trên bầu trời.

**LINEAR REGRESSION (hồi quy tuyến tính - LR)** là model địa tâm trong thống kê. LR ở đây là một nhóm golem thống kê đơn giản để tìm mean và variance trong data bằng cách dùng phép cộng. Giống thuyết địa tâm, LR có thể mô tả rất nhiều hiện tượng tự nhiên. LR là model mô tả tương ứng với nhiều model mô tả khác nhau. Nếu chúng ta phân tích từng cấu trúc của nó, có thể ta sẽ sai lầm. Nhưng nếu dùng tốt, những con golem tuyến tính cũng có ích.

Chương này giới thiệu LR kiểu Bayes. Dưới sự diễn giải bằng xác suất, tức là công việc của một Bayesian, LR dùng phân phối Gaussian (phân phổi normal - bình thường) để mô tả tính bất định của sự đo đạc. Model này đơn giản, linh hoạt và được dùng rất nhiều. Cũng giống tất cả model thống kê, LR không áp dụng cho mọi trường hợp. Nhưng nó là model cơ bản nhất, vì nếu bạn hiểu nó, bạn có thể dễ dàng tiếp cận những dạng regress khác ít phổ biến hơn. 


## <center>4.1 Tại sao phân phối normal lại normal</center><a name="1"></a>

Giả sử bạn có 1000 người xếp hàng ở đường giữa sân bóng. Mỗi người có 1 đồng xu. Mỗi lần huýt sáo thì họ sẽ lần lượt tung đồng xu. Nếu đồng xu ra mặt ngửa, người đó qua bên trái đường giữa một bước. Nếu đồng xu ra mặt sấp, người đó qua bên phải đường giữa một bước. Mỗi người thực hiện tung đồng xu 16 lần, và đứng yên sau đó. Giờ chúng ta đo khoảng cách từ mỗi người đến đường giữa, bạn có thể đoán được tỉ lệ nào trong 1000 người đó nằm ở đường giữa không? Còn 10m bên trái đường giữa thì sao?

Rất khó để biết cụ thể một người nào sẽ đứng ở đâu, nhưng bạn có thể tự tin và nói rằng tại vị trí nào đó sẽ chiếm tỉ lệ bao nhiêu. Số đo khoảng cách ấy sẽ phân phối theo phân phối normal, hay Gaussian. Điều này đúng ngay khi phân phối gốc là binomial, bởi vì có vô số khả năng xảy ra để một trình tự trái phải mà tổng là zero. Những trình tự ra một bước trái hoặc phải thì ít hơn, và cứ thế với những trình tự còn lại. Cuối cùng sẽ ra phân phối normal với hình chuông đặc trưng.

### 4.1.1 Normal qua phép cộng

Ta có thể mô phỏng quy trình trên bằng cách tạo cho mỗi người một dãy 16 số gồm -1 và 1 một cách ngẫu nhiên, ta tính tổng chúng lại, và lặp lại cho 1000 người. 

```python
pos = np.sum(dist.Uniform(-1, 1).sample(PRNGKey(0), (1000, 16)), -1)
```

![16step](/assets/images/figure 4-2.svg)

Hình bên trái biểu diễn vị trí của 1000 người ở bước thứ 4, 8, 16. Và ở vị trí 16 thì phân bố các vị trí ấy sẽ gần giống với phân phối normal, như hình bên phải. Mặc dù phân phối ban đầu có vẻ rời rạc hỗn loạn, sau 16 bước, nó gần như có hình chuông quen thuộc. Hình chuông ấy của phân phối normal được xuất phát từ sự ngẫu nhiên. Bạn có thể thí nghiệm với số bước nhiều hơn để kiểm tra phân phối khoảng cách có tuân theo phân phối Gaussian hay không. Bạn có thể bình phương khoảng cách giữa 2 bước chân hay biến đổi các số tuỳ ý, kết quả vẫn không thay đổi: Phân phối normal vẫn xuất hiện. Tại sao?

Mọi quy trình có phép cộng tất cả các giá trị ngẫu nhiên từ chung một phân phối đều ra phân phối normal. Nhưng không dễ để nắm bắt được tại sao phép cộng lại ra đường cong của tổng. Bạn có thể suy nghĩ như vậy: Cho dù giá trị trung bình của phân phối gốc là gì, mỗi lần lấy mẫu từ nó có thể nghỉ là sự dao động của con số trung bình ấy. Dao động dương lớn sẽ bù trừ dao động âm. Số các phần tử trong tổng càng nhiều, thì xác suất để mỗi dao động bị bù trừ bởi dao động khác, hoặc tập hợp nhiều dao động nhỏ đối dấu hơn. Cho nên tổng số cuối cùng mà có khả năng nhất, là tổng mà mọi dao động đều bị bù trừ, hay tổng là zero (tương quan với trung bình).

Không quan trọng hình dáng của phân phối nền của quy trình lấy mẫu. Nó có thể là uniform, như ví dụ ở trên, hoặc cũng có thể bất kf phân phối nào. Tuỳ vào loại phân phối nền mà có thể hội tụ chậm, nhưng nó phải xảy ra. Thông thường, như ví dụ, sự hội tự diễn ra rất nhanh.

### 4.1.2 Normal qua phép nhân

Đây là một cách khác để có phân phối normal. Giả sử tốc độ phát triển của vi khuẩn bị ảnh hưởng bởi vài loci trong gen, những loci này chứa allel mã hoá sự phát triển. Giả sử tất cả những gen này tương tác với nhau, ví dụ như mỗi gen làm tăng thêm phần trăm sự phát triển. Có nghĩa là hiệu ứng của chúng là phép nhân hơn là phép cộng .

```python
np.prod(1 + dist.Uniform(0, 0.1).sample(PRNGKey(0), (12,)))
```

Code trên tạo 12 con số từ 1.0 đến 1.1, mỗi số tương ứng với tỉ lệ phát triển. Như 1.0 là không phát triển và 1.1 là tăng 10%. Tích của chúng sẽ phân phối theo normal. Thật vậy ta có thể lấy 1000 con số như vậy và kiểm tra.

```python
growth = np.prod(1 + dist.Uniform(0, 0.1).sample(PRNGKey(0), (1000, 12)), -1)
az.plot_dist(growth)
x = np.sort(growth)
plt.plot(x, np.exp(dist.Normal(np.mean(x), np.std(x)).log_prob(x)), "--");
plt.title('growth');
```

![](/assets/images/figure 4-3.svg)

Phân phối normal xuất hiện từ tổng các dao động ngẫu nhiên, nhưng hiệu ứng của mỗi loci là phép nhân. Tại sao?

Chúng ta lần nữa hồi tụ về phân phối normal, bởi vì hiệu ứng nhân tại mỗi loci là quá nhỏ. Phép nhân số nhỏ có thể ước lượng giống như phép cộng. Ví dụ như 2 loci tăng sự phát triển 10%, thì tích là:

$$ 1.1 \times 1.1 =1.21 $$

Ta có thể ước lượng số này bằng phép cộng số trên, và lệch khoảng 0.01:

$$ 1.1 \times 1.1 = (1+0.1)(1+0.1) = 1 +0.2 +0.01 \approx 1.2 $$

Hiệu ứng của loci càng nhỏ, thì ước lượng bằng phép cộng càng tốt. Bằng cách này, hiệu ứng nhỏ nhân với nhau cũng giống như phép cộng, và chúng thường ổn định dần thành phân phối Gaussian. Bạn có thể kiểm tra lại:

```python
big = np.prod(1 + dist.Uniform(0, 0.5).sample(PRNGKey(0), (1000, 12)), -1)
small = np.prod(1 + dist.Uniform(0, 0.01).sample(PRNGKey(0), (1000, 12)), -1)
```

Độ lệch trong sự phát triển có tương tác, chỉ cần nó đủ nhỏ, sẽ hội tụ thành phân phối Gaussian.

### 4.1.3 Normal qua log của phép nhân

Chưa hết, với độ lệch lớn trong phép nhân mà không tạo ra phân phối Gaussian, nó có thể là Gaussian với thang đo của logarit.

```python
log_big = np.log(np.prod(1 + dist.Uniform(0, 0.5).sample(PRNGKey(0), (1000, 12)), -1))
```

Nó là một phân phối Gaussian. Chúng ta có phân phối Gaussian bởi vì phép cộng trong log tương ứng với phép nhân. cho nên tương tác nhân với độ lệch lớn vẫn có thể tạo ra phân phối Gaussian, nếu chúng ta đo lường kết quả ở thang đo của log. Vì đo lường là ngẫu nhiên, cho nên không có gì nghi ngại với sự biến đổi này. Dù sao đi nữa, người ta vẫn đo lượng âm thanh và động đất, thậm chí thông tin bằng thang đo log.

### 4.1.4 Sử dụng phân phối Gaussian

Chúng ta sẽ sử dụng phân phối Gausssian xuyên suốt chương này để tạo bộ khung cho giả thuyết của chúng ta, và xây dựng model đo lường như phép cộng của phân phôi normal. Lý giải về tại sao dùng phân phối này gồm 2 nhóm: (1) tự nhiên và (2) phương pháp học.

Theo lý giải tự nhiên, thế giới này chứa rất nhiều phân phối Gaussian một cách tương đối. Chúng ta không thể nào trải nghiệm phân phối Gaussian hoàn hảo. Nhưng nó tồn tại nhiều nơi, với độ lớn khác nhau và trong lĩnh vực khác nhau. Sai số đo lường, biến thiên trong sự phát triển, tốc độ của nguyên tử luôn hội tụ về phân phối Gaussian. Tại sao? Vì chúng chẳng qua là phép cộng các sự dao động. Và phép cộng dao động xảy ra vô hạn sẽ cho kết quả là phân phối của các tổng chứa tất cả thông tin của quy trình bên dưới, nằm ngoài con số trung bình và độ lan rộng.

Một trong những hậu quả của model thống kê dùng phân phối Gaussian là nó không phát hiện tốt những quy trình siêu nhỏ. Nó gợi nhớ lại triết lý model ở chương 1. Nhưng nó cũng có nghĩa là có thể làm được nhiều việc tốt. Nếu chúng ta biết được sinh học phát triển của chiều cao trước khi ta làm model thống kê cho chiều cao, sinh học con người sẽ không tiến bộ.

Có rất nhiều mô hình trong tự nhiên, cho nên đừng nghĩ rằng phân phối Guassian áp dụng được cho mọi thứ. Các chương sau ta sẽ gặp các mô hình như exponential, gamma, Poisson đều có từ tự nhiên. Phân phối Gaussian là một thành viên trong tập hợp phân phối tự nhiên cơ bản, còn gọi là **EXPONENTIAL FAMILY**. Tất cả các thành viên của tập hợp này đêu rất quan trọng cho khoa học, bởi vì nó tạo ra thế giới của chúng ta.

Nhưng sự xuất hiện tự nhiên của phân phối Gaussian chỉ là một trong nhiều lý do để xây model dựa trên nó. Theo lý giải phương pháp học, Gaussian đại diện cho một trạng thái thiếu hiểu biết. Nếu những gì chúng ta biết về phân phối của đo lường là trung bình và phương sai, thì phân phối Gaussian phù hợp nhất với giả định của chúng ta.

Nói cách khác là phân phối Gaussian là sự thể hiện tự nhiên nhất về trạng thái thiếu hiểu biết của chúng ta, bởi vì nếu giả định rằng đo lường có phương sai giới hạn, phân phối Gaussian là hình dạng có thể biểu diễn số lượng các cách lớn nhất mà không có giả định mới được đưa vào. Nó ít bất ngờ nhất và cần ít thông tin giả định nhất. Bằng cách này, phân phối Gaussian là phù hợp nhất với giả định của golem. Nếu bạn không nghĩ phân phối là Gaussian, bạn nên nói rõ bạn biết gì để nói cho golem, việc này sẽ cải thiện kết quả suy luận.

Lý giải phương pháp học này là mở đầu của **INFORMATION THEORY** và **MAXIMUM ENTROPY**. Các chương sau ta sẽ gặp nhưng phân phối thường gặp và hữu ích khác để xây *generalized linear model(GLM)*. Với chúng, ta có thể thêm áp chế vào model để làm nó thích hợp hơn.

Hãy nhớ rằng ta không cần phải thề thốt gì với model cả. Golem là người hầu của chúng ta, không phải gì đó khác.

>**Nghĩ lại: Hai đuôi lớn.** Gaussian thường gặp trong tự nhiên và có một số tính chất rất đẹp. Nhưng dùng nó phải chấp nhận một số nguy cơ. Hai đuôi của phân phối Gaussian rất nhỏ - vấn đề thường không ảnh hưởng nhiều. Thực ra thì phần lớn mật độ của Gaussian nằm trong giới hạn 1 độ lệch chuẩn quanh trung bình. Nhiều quy trình tự nhiên và không tự nhiên có hai đuôi lớn hơn, có nghĩa là có xác suất tạo ra giá trị extreme cao hơn. Ví dụ thực tế và quan trọng là financial time series - sự kiện lên và xuống của chứng khoán có thể nhìn giống Gaussian trong thời gian ngắn, nhưng khi thời gian dài ra, cú shock extreme làm cho model Gaussian không hoạt động được. Historical time series cũng tương tự, và suy luận ví dụ như xu hướng chiến tranh có thể bị bất ngờ tại hai đuôi lớn. Ta có một số thay thế cho Gaussian.

---

**Nghĩ nhiều hơn: phân phối Gaussian.** Bạn không cần phải nhớ phân phối xác suất Gaussian. Máy tính đã biết nó. Nhưng vài kiến thức về hình dạng của nó cũng có ích. Hàm mật độ xác suất (**Probability density**) của giá trị *y*, cho rằng phân phối Gaussian có trung bình $\mu$ và độ lệch chuẩn $\sigma$: 

$$ p(y|\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} exp \left( -\frac{(y-\mu)^2}{2\sigma^2} \right) $$

Nó thật khủng. Phần quan trọng là ở $(y-\mu)^2$, nó cho ta có dạng hình cong parabol. Khi ta dùng *e* luỹ thừa parabol thì ra hình chuông cổ điển. Phần còn lại là scale và chuẩn hoá phân phối.

Gaussian là phần phối liên tục, không phải phần phối rời rạc như bài trước. Hàm xác suất với kết quả rời rạc, như binomial, gọi là hàm *probability mass* và ký hiệu Pr. Phân phối liên tục như Gaussian gọi là *probability density* và ký hiệu *p* và *f*. Vì lý do toán học, mật độ xác suất có thể lớn hơn 1. Bạn có thử tính *p*(0 \| 0, 0.1). Kết quả gần bằng 4. Mật độ xác suất là tốc độ thay đổi của xác suất tích luỹ. Cho nên xác suất tích luỹ tăng nhanh, thì mật độ có thể hơn 1. Nhưng nếu ta tính diện tích dưới hàm mật độ, nó không bao giờ hơn 1. Phần diện tích đó cũng có thể gọi là khối lượng (mass). Chúng ta thường không quan tâm mật độ hay khối lượng khi dùng tính toán. Nhưng vẫn tốt khi để ý sự khác biệt này, đôi khi khác biệt vẫn có ảnh hưởng.

Đôi khi bạn có thể thấy Gaussian với parameter $\tau$ (tau) thay vì $\sigma$. $\tau$ thường gọi là *độ chính xác* và bằng 1/$\sigma^2$. Khi $\sigma$ lớn, $\tau$ nhỏ.  Công thức density khi đó là:

$$ p(y|\mu,\tau) = \sqrt{\frac{\tau}{2\pi}}exp\left( -\frac{1}{2}\tau(y-\mu)^2  \right) $$

Một số phần mềm Bayes như BUGS hoặc JAGS dùng $\tau$ hơn là $\sigma$.

---

## <center>Ngôn ngữ mô tả model</center><a name="2"></a>

Sách này dùng một tiêu chuẩn ngôn ngữ để mô tả và code model thống kê. Bạn sẽ gặp loại ngôn ngữ này trong nhiều sách thống kê và tất cả tạp chí thống kê, vì nó được dùng chung cho model Bayes và non-Bayes. Các nhà khoa học ngày càng nhiều dùng loại ngôn ngữ này để mô tả phương pháp thống kê của họ. Cho nên học ngôn ngữ này là một sự đầu tư.

Cách tiếp cận như sau:

1. Liệt kê các variables mà ta cần làm việc. Có variable quan sát được, gọi là data. Những variable không quan sát được, gọi là parameter.
2. Ta định nghĩa nhưng variable dưới dạng variable khác hoặc phân phối xác suất.
3. Sự kết hợp của variable và các phân phối xác suất gọi là *joint generative model* có thể dùng cho mô phỏng quan sát giả thuyết cũng như phân tích data thật.

Mẫu tiếp cận này áp dụng cho model ở mọi lĩnh vực, từ thiên văn đến lịch sử nghệ thuật. Khó khăn lớn nhất nằm ở câu chuyện - Variable nào quan trọng và giả thuyết làm thế nào để liên kết chúng - chứ không phải công thức toán học.

Sau khi quyết định các var và liên kết, ta trình bày như sau:

$$ \begin{matrix}
y & \sim \text{Normal} (\mu_i, \sigma) \\
\mu_i & = \beta x_i \\
\beta & \sim \text{Normal} (0,10) \\
\sigma & \sim \text{Exponetial}(1)\\
x_i & \sim \text{Normal}(0,1)
\end{matrix}$$

Nếu bạn không hiểu gì, tốt. Nó có nghĩa là bạn đang đọc đúng sách, bởi vì sách này sẽ dạy bạn cách đọc và viết mô tả những model thống kê. Ta không cần tính toán nào cho model. Thay vào đó, nó cho ta định nghĩa rõ ràng và nói chuyện với model.

Cách tiếp cận trên không phải cách duy nhất để mô tả model, nhưng nó là ngôn ngữ phổ biến và hiệu quả. Một khi nhà khoa học hiểu được ngôn ngữ này, thì họ sẽ có thể liên kết dễ dàng với giả định của model. Ta không cần phải nhớ những điều kiện test khủng khiếp như cùng một variance, vì ta có thể đọc chúng từ mô tả model. Ta có thể nhìn thấy cách tự nhiên để thay đổi giả định, thay vì bị giới hạn lại như model hồi quy, ANOVA, ANCOVA. Chúng là chung một model, nhưng ta chỉ biết nếu ta đọc được model và kết nối variable với phân phối trên một tập variable khác.

### 4.2.1 Quay lại ví dụ model tung quả cầu

Ta có thể mô tả model như sau:

$$ \begin{matrix}
W & \sim \text{Binomial} (N, p) \\
p & \sim \text{Uniform} (0,1) \\
\end{matrix}$$

W là số đếm nước, N là số lần tung, và *p* là tỉ lệ bề mặt nước.

Khi ta hiểu model theo cách này, ta tự động hiểu được mọi giả định của nó. Ta biết phân phối binomial giả định mỗi mẫu đều độc lập với nhau, cho nên ta biết model giả định mỗi lần tung độc lập với nhau.

Ta sẽ tập trung vào những model đơn giản. Ở model trên, dòng đầu tiên định nghĩa hàm likelihood trong Bayes' theorem. Những dòng khác định nghĩa prior. Những dòng này là **STOCHASTIC**, tức phân phối ngẫu nhiên, ký hiệu bằng $\sim$. Quan hệ stochastic nghĩa là một ánh xạ của variable hoặc parameter tới một phân phối. Nó là stochastic vì không một giá trị nào của biến được biết cụ thể. Nó là xác suất: vài giá trị thì có xác suất xuất hiện cao hơn, nhưng rất nhiều giá trị khác vẫn có xác suất xảy ra. Sau này, ta sẽ gặp model có những định nghĩa khẳng định trong nó.

---

**Nghĩ nhiều hơn: Từ định nghĩa model đến Bayes' theorem.** Ta có thể dùng Bayes' theorem để viết model như sau:

$$Pr(p|w,n) = \frac{\text{Binomial}(w|n,p) \text{Uniform}(p|0,1)}{\int \text{Binomial}(w|n,p) \text{Uniform}(p|0,1) dp} $$

Mẫu số khủng này là xác suất trung bình đã nói. Nó chuẩn hoá posterior để tổng bằng 1. Tử số là posterior tỉ lệ thuận với prior nhân với likelihood. Nó giống như grid approx mà ta đã làm.

---

## <center>Model Gaussian chiều cao</center><a name="3"></a>

Ta sẽ xây model linear regression. Nó sẽ là "regression" một khi có predictor variable. Trước mắt ta sẽ để trống nó để cho phần sau. Hiện tại, ta dùng một variable đo lường trong model như một phân phối Gaussian. Có 2 parameter để mô tả hình dạng của phân phối: trung bình $\mu$ và độ lệch chuẩn $\sigma$. Bayesian updating sẽ cho ta xem xét mọi khả năng của $\mu$ và $\sigma$ với data tạo ra posterior.

Nói một cách khác, có vô hạn phân phối Gaussian, có phân phối với trung bình nhỏ, trung bình lớn, có phân phối rộng với $\sigma$ lớn, hoặc hẹp hơn. Ta muốn cỗ máy Bayes xem xét mọi khả năng xảy ra với mọi phân phối với $\mu$ và $\sigma$, xếp hạng chúng bởi posterior. Posterior cho phép một cách đo lường sự thích hợp logic với mỗi khả năng phân phối của data và model.

Trong thực hành, ta thường dùng ước lượng để phân tích. Chúng ta không thực sự xem xét mọi trường hợp của $\mu$ và $\sigma$. Nhưng ta cũng không tốn gì trong đa số trường hợp. Điều cần nhớ là ta hiểu rằng "ước lượng" là toàn bộ posterior chứ không phải một điểm nào trong đó. Kết quả là, phân phối posterior sẽ là phân phối của phân phối Gaussian. Đúng, phân phối của phân phối. Nếu bạn không hiểu, có nghĩa bạn là người trung thực. Bạn sẽ hiểu sớm thôi.

### 4.3.1 Data

Ta sẽ dùng data Howell1 dân số của Dobe area !Kung San, được thực hiện bởi Nancy Howell từ năm 1960. 

```python
Howell1 = pd.read_csv("../data/Howell1.csv", sep=";")
d = Howell1
d.info()
# -> có 352 dòng
```

```python
d.head()
```

|height|  weight|  age| male|
|-|-|-|-|
|0|   151.765| 47.825606|   63.0|    1|
|1|   139.700| 36.485807|   63.0|    0|
|2|   136.525| 31.864838|   65.0|    0|
|3|   156.845| 53.041915|   41.0|    1|
|4|   145.415| 41.276872|   51.0|    0|

Ta có thể tổng quan data bằng:

```python
print_summary(dict(zip(d.columns, d.T.values)), 0.89, False)
```

Ta sẽ làm việc với column Height, với độ tuổi >18t, vì chiều cao tương quan rất lớn với trẻ nhỏ.

```python
d2 = d[d.age >= 18]
```

### 4.3.2 Model

Mục tiêu là model những giá trị này bằng phân phối Gaussian. Trước tiên, ta sẽ vẽ phân phối của height. Nó có vẻ giống Gaussian. Bởi vì chiều cao là tổng của nhiều yếu tố phát triển. Như đã đề cập ở đầu chương, phân phối của tổng thường là hội tự về phân phối Gaussian. Dù lý do thế nào, chiều cao người lớn của một quần thể thường là normal.

Đó là những lý do để dùng phân phối Gaussian. Nhưng hãy cẩn thận nếu chọn Gaussian khi hình vẽ phân phối giống Gaussian. Chỉ nhìn vào data thô, mà quyết định cách model nó, không phải ý tưởng tốt. Data có thể là hỗn hợp của nhiều phân phối Gaussian, lúc đó nhìn bằng mắt thường sẽ không thấy được phân phối normal của nó. Hơn nữa, như đã nói ở lúc đầu, phân phối quan sát được không nhất thiết phải Gaussian để dùng phân phối Gaussian.

Cố vô số phân phối Gaussian, với vô số trung bình và độ lệch chuẩn. Model được định nghĩa:

$$ h_i \sim \text{Normal} (\mu, \sigma) $$

Có sách ghi là $h_i \sim \mathcal{N}(\mu,\,\sigma) $, nó là cùng một model với cách ghi khác. Ký hiệu *h* chỉ data chiều cao, và *i* nhỏ là vị trí trong data, có nghĩa là *index*. Và trong data này là *i* có giá trị từ 1 đến 352. Với định nghĩa này, golem sẽ hiểu mỗi số đo lường chiều cao được định nghĩa từ chung một phân phối normal với trung bình $\mu$ và $\sigma$. Sau này, bạn sẽ thấy ý nghĩa của *i* nhỏ, nó sẽ được thể hiện ở bên phải của định nghĩa model, cho nên đừng quên để ý nó, mặc dù hiện tại nó có vẻ vô nghĩa.

>**Nghĩ lại: phân phối độc lập và giống nhau.** Model ở trên giả định $h_i$ là phân phối độc lập và giống nhau (independent and identically distributed, i.i.d, iid, hoặc IID), bạn có thể gặp model trên được viết:  
$$ h_i \stackrel{i.i.d.}{\sim} \text{Normal} (\mu, \sigma) $$  
"iid" chỉ rằng mỗi giá trị $h_i$ đều có chung hàm xác suất, độc lập với những giá trị $h$ khác và sử dụng chung parameter. Ta có cảm giác điều này có vẻ không đúng. Ví dụ, chiều cao trong chung gia đình thường sẽ tương quan bởi vì có chung allel trong dòng họ.  
Giả định iid không có gì sai, chỉ cần nhớ rằng nó nằm trong golem trong thế giới nhỏ, không phải thế giới lớn. Giả định iid về cách golem đại diện cho tính bất định của nó. Đây là một giả định về mặt *phương pháp học*. Nó không phải giả định thực thể về thế giới, về mặt *tự nhiên*. E.T.Jaynes (1922-1998) gọi đây là *sự lừa đảo do ánh xạ tâm trí*, lỗi lầm do hiểu sai lý do về phương pháp học và lý do tự nhiên. Trọng điểm ở đây là không phải phương pháp học tốt hơn tự nhiên, nhưng với sự hiểu biết về mối tương quan này, có lẽ "iid" là phân phối tốt nhất. Hơn nữa, có một kết quả toán học, *de Finetti's theorem*, nói rằng các giá trị là **EXCHANGABLE** có thể được ước lượng từ hỗn hợp nhiều phân phối iid. Nói dễ hiểu thì giá trị exchangeable có thể tái sắp xếp được. Tác động thực tế của iid không thể hiểu theo nghĩa đen. Có một vài loại tương quan có thể thay đổi một ít về hình dạng của phân phối, chúng ảnh hưởng đến trình tự xuất hiện của giá trị. Ví dụ, cặp đôi có tương quan cao về chiều cao. Nhưng phân phối chung của chiều cao vẫn normal. MCMC lợi dụng điều này, thường sử dụng những trình tự mẫu có tương quan cao để ước lượng mọi phân phối ta thích.

Để hoàn thành model, ta cần priors. Parameter cần ước lượng là $\mu$ và $\sigma$, cho nên ta cần prior Pr($\mu,\sigma$), hay xác suất prior kết hợp của toàn bộ parameters. Trong nhiều trường hợp, prior thường được định nghĩa cụ thể độc lập cho mỗi parameter, giả định Pr($\mu,\sigma$) = Pr($mu$)Pr($\sigma$):

$$ \begin{matrix}
h_i & \sim \text{Normal} (\mu, \sigma) && \quad \; [\text{likelihood}]\\
\mu & \sim \text{Normal} (178,20) && \quad [\mu \; \text{prior}]\\
\sigma & \sim \text{Uniform} (0, 50) && \quad [\sigma \; \text{prior}]
\end{matrix} $$

Prior của $\mu$ là một prior Gaussian khá rộng, trung bình ở 178 cm, với xác suất 95% giữa 178 $\pm$ 40 cm.

Tại sao 178 cm? Tác giả sách cao 178 cm. Khoảng cách từ 138 cm đến 218 cm đảm bảo một khoảng lớn chiều cao ở dân số loài người. Vậy thông tin kiến thức chung đã có trong prior. Mọi người đều biết gì đó về chiều cao con người và đặt một prior đúng logic cho đại lưọng này. Nhưng trong nhiều vấn đề regression, sử dụng thông tin prior rất khó vì parameter nhiều khi không có ý nghĩa thực thể.

Cho dù prior là gì, thì ta vẫn nên vẽ prior ra, để có được một tầm nhìn về giả định mà ta đưa vào model.

```python
x = np.linspace(100, 250, 101)
plt.plot(x, np.exp(dist.Normal(178, 20).log_prob(x)));
```

![](/assets/images/figure 4-4.svg)

Bạn sẽ thấy rằng golem giả định chiều cao trung bình là nằm trong khoảng 140 cm và 220 cm. Có nghĩa là prior này chứa ít thông tin chứ không nhiều. Prior $\sigma$ là prior phẳng, có giá trị từ 0 đến 50 cm, vì sigma luôn luôn dương. Làm sao chọn giới hạn trên? Độ lệch chuẩn 50cm sẽ cho rằng 95% chiều cao cá nhân sẽ nằm trong 100cm trên dưới chiều cao trung bình. Và nó là khoảng lớn.

Câu chuyện nãy giờ là tốt. Nhưng tốt hơn khi ta nhìn thấy prior sẽ cho phân phối chiều cao cá nhân như thế nào. **PRIOR PREDICTIVE** là một phần quan trọng của model. Khi bạn chọn xong prior cho *h*, $\mu$ và $\sigma$, tạo ra phân phối xác suất kết hợp joint distribution cho chiều cao cá nhân. Bằng cách mô phỏng, ta sẽ thấy lựa chọn của bạn sẽ cho quan sát như thế nào. Điều này sẽ giúp bạn chẩn đoán lựa chọn xấu. Rất nhiều lựa chọn thuận tiện cho kết quả xấu, và ta có thể thấy nó thông qua mô phỏng prior predictive.

```python
sample_mu = dist.Normal(178, 20).sample(PRNGKey(0), (int(1e4),))
sample_sigma = dist.Uniform(0, 50).sample(PRNGKey(1), (int(1e4),))
prior_h = dist.Normal(sample_mu, sample_sigma).sample(PRNGKey(2))
az.plot_kde(prior_h, bw=1);
```

![](/assets/images/figure 4-5.svg)

Hình trên cho thấy mật độ hình chuông với 2 đuôi lớn. Đây là phân phối của chiều cao mong đợi, trung bình hoá trên prior. Chú ý rằng phân phối xác suất prior chiều cao không phải normal. Điều này không sao. Phân phối bạn thấy không phải là mong đợi theo kinh nghiệm, nhưng là phân phối các khả năng có thể có của height, trước khi thấy data.

Mô phỏng prior predictive có thể tốt để gán các prior hợp lý, cho nên rất khó để đánh giá prior ảnh hưởng thế nào đến biến quan sát được. Ví dụ xem xét prior ít thông tin hơn cho $\mu$, như $\mu \sim \text{Normal} (178, 100)$. Priors như vậy rất thường gặp trong model Bayes, nhưng không hợp lý. 

```python
sample_mu = dist.Normal(178, 100).sample(PRNGKey(0), (int(1e4),))
prior_h = dist.Normal(sample_mu, sample_sigma).sample(PRNGKey(2))
az.plot_kde(prior_h, bw=1);
```

![](/assets/images/figure 4-6.svg)

Kết quả cho thấy tồn tại chiều cao là số âm. Nó cũng có nhiều số khổng lồ. Trong lịch sử loài người, người cao nhất là Robert Pershing Wadlow (1918-1940) cao 272 cm. Trong mô phỏng của chúng ta, 18% người cao hơn chiều cao này.

Có ảnh hưởng không? Trong trường hợp này, ta có nhiều data nên prior ngu ngốc là không gây hại. Nhưng không phải lúc nào cũng vậy. Có rất nhiều câu hỏi suy luận mà chỉ data là không đủ, cho dù nhiều cỡ nào. Bayes cho chúng ta tiếp tục trong những trường hợp này. Nhưng khi ta dùng kiến thức khoa học để tạo prior hợp lý. Sử dụng kiến thức khoa học để tạo prior không phải gian lận. Quan trọng ở đây là prior không phải đựa trên giá trị trong data, mà là những gì bạn biết trước khi thấy data.

>**Nghĩ lại: Tạm biệt epsilon.** Nhiều bạn sẽ thấy model linear như sau:  
$$ \begin{matrix}
h_i & = \mu + \epsilon_i \\
\epsilon_i & \sim \text{Normal}(0, \sigma)
\end{matrix} $$  
Điều này đồng nghĩa với $h_i \sim \text{Normal}(\mu,\sigma)$, với $\epsilon$ là mật độ Gaussian. Nhưng dạng này không tốt. Nó không tổng quát được cho các loại model khác. Nó không thể nào diễn đạt model non-Gaussian bằng $\epsilon$.

---

**Nghĩ nhiều hơn: quay lại Bayes' theorem.** Định nghĩa trên viết dưới dạng Bayes' theorem:

$$ Pr(\mu,\sigma|h) = \frac{\Pi_i\text{Normal}(h_i|\mu,\sigma)\text{Normal}(\mu|178,20)\text{Uniform}(\sigma|0,50)}{\int\int\Pi_i\text{Normal}(h_i|\mu,\sigma)\text{Normal}(\mu|178,20)\text{Uniform}(\sigma|0,50)d\mu d\sigma} $$

Nó thật khủng khiếp, nhưng nó cũng giống như ở trên. 2 thứ mới làm cho nó trở nên khủng. Thứ 1 là có nhiều hơn một quan sát của *h*, để có được likelihood của toàn bộ data, ta phải tính xác suất mỗi *h* và nhân chúng lại với nhau. Thứ 2 là có 2 priors, $\mu$ và $\sigma$, và chúng dồn lại. Trong grid approx sau đây, bạn sẽ thấy sự ứng dụng của định nghĩa này. Mọi thứ được tính trên thang đo log (log scale), nên phép nhân thành phép cộng.

---

### 4.3.3 Grid approx cho phân phối posterior

Bởi vì đây là model đầu tiên có 2 parameters, nó đáng để ta dùng tính posterior bằng vũ lực. Mặc dù tôi không khuyên dùng ở nơi khác, bởi vì nó rất vất vả và nhiều phép tính. Thực ra, nó không thực dụng và đôi lúc bất khả thi. Cũng giống như mọi khi, nó đáng để biết mục tiêu nhìn như thế nào, trước khi chấp nhận các hình dạng ước lượng. Sau đó, ta sẽ dùng quadratic approx để ước tính posterior, và có thể so sánh kết quả từ 2 phương pháp ước lượng.

```python
mu_list = np.linspace(start=150, stop=160, num=100)
sigma_list = np.linspace(start=7, stop=9, num=100)
mesh = np.meshgrid(mu_list, sigma_list)
post = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
post["LL"] = vmap(lambda mu, sigma: np.sum(dist.Normal(mu, sigma).log_prob(
    d2.height.values)))(post["mu"], post["sigma"])
logprob_mu = dist.Normal(178, 20).log_prob(post["mu"])
logprob_sigma = dist.Uniform(0, 50).log_prob(post["sigma"])
post["prob"] = post["LL"] + logprob_mu + logprob_sigma
post["prob"] = np.exp(post["prob"] - np.max(post["prob"]))
```

Cách làm trên cũng tương tự như bài trước, nhưng ta thực hiện trên log scale, phép nhân sẽ thành phép cộng, kết quả posterior được scale lại bằng max, nếu không sẽ ra giá trị zero do tính năng làm tròn của phần mềm. Kết quả là những xác suất tương đối (không phải chính xác), nhưng nó cũng đủ để mô tả posterior.

Bạn có thể vẽ biểu đồ thể hiện posterior qua các lệnh sau:

```python
plt.contour(post["mu"].reshape(100, 100), post["sigma"].reshape(100, 100),
            post["prob"].reshape(100, 100));
```

![](/assets/images/figure 4-7.svg)

```python
plt.imshow(post["prob"].reshape(100, 100),
           origin="lower", extent=(150, 160, 7, 9), aspect="auto");
```

![](/assets/images/figure 4-8.svg)

### 4.3.4 Lấy mẫu từ posterior

Để tìm hiểu posterior này chi tiết hơn, ta sẽ lấy mẫu từ nó như chương 3. Cái mới ở đây là ta có 2 parameter, và ta cần lấy mẫu từ 2 phân phối. Đầu tiên ta chuẩn hoá posterior, lấy 1000 mẫu từ posterior, và lấy mẫu mu và sigma tương ứng với posterior.

```python
prob = post["prob"] / np.sum(post["prob"])
sample_rows = dist.Categorical(probs=prob).sample(PRNGKey(0), (int(1e4),))
sample_mu = post["mu"][sample_rows]
sample_sigma = post["sigma"][sample_rows]
```

Ta có 10,000 mẫu có tái chọn lại, từ posterior của height data.

```python
plt.scatter(sample_mu, sample_sigma, s=64, alpha=0.1, edgecolor="none");
```

![](/assets/images/figure 4-9.png)

Như chương 3, ta có thể mô tả phân phối posterior theo $\mu$ và $\sigma$.

```python
az.plot_dist(sample_mu)
az.plot_dist(sample_sigma);
```

![](/assets/images/figure 4-10.svg)

Đây là phân phối biên của mu và sigma, "biên (marginal)" nghĩa là trung bình trên những parameters khác. Kết quả hình vẽ cho thấy no khá giống normal. Khi cỡ mẫu tăng lên, mật độ posterior tiến dần đến phân phối normal. Nếu bạn nhìn kỹ, mật độ $\sigma$ có đuôi dài hơn. Tình trạng này rất thường gặp ở parameter $\sigma$. Điều này dễ hiểu bởi vì $\sigma$ luôn dương, nên tính bất định sẽ hướng về variance lớn như thế nào hơn là variance nhỏ.x

Để mô tả mật độ, ta dùng HPDI hoặc mean, median, quantiles:

```python
hpdi(sample_mu, 0.89)
hpdi(sample_sigma, 0.89)
# [153.93939 155.15152]
# [7.3030305 8.232324 ]
```

### 4.3.5 Tìm posterior distribution bằng quadratic approx

Giả định đỉnh của posterior nằm ở **MAXIMUM A POSTERIORI (MAP)**, thì ta có thể có hình dạng của posterior bằng phương pháp quad approx đỉnh của posterior.

Để sử dụng phương pháp này, đầu tiên ta định nghĩa model như phần trước. Sau đó dùng thuật toán để leo lên posterior từ từ và tìm đỉnh MAP của nó. Dựa vào đỉnh, ta ước lượng bằng quad approx để tạo phân phối posterior. Nhớ rằng, cách làm này giống như quy trình của non-Bayes, chỉ ngoại trừ Prior.

Data được định nghĩa như sau:

$$ \begin{matrix}
h_i & \sim \text{Normal}(\mu, \sigma)\\
\mu & \sim \text{Normal}(178, 20)\\
\sigma &\sim \text{Uniform}(0, 50)
\end{matrix} $$

Code numpyro tương ứng:

```python
def flist(height):
    mu = numpyro.sample("mu", dist.Normal(178, 20))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
```

Sau đó ta sẽ fit vào `AutoLapaceApproximation` là function tìm MAP của numpyro, SVI sẽ chạy leo lên đỉnh của posterior. SVI nhận arg là: model, guide, optimizer, loss, \*kwargs của model.

```python
m4_1 = AutoLaplaceApproximation(flist)
svi = SVI(flist, m4_1, optim.Adam(1), AutoContinuousELBO(),
          height=d2.height.values)
init_state = svi.init(PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, np.zeros(2000))
p4_1 = svi.get_params(state)
```

Sau khi chạy SVI 2000 lần thì ta có posterior, ta có thể mô tả nó như sau:

```python
samples = m4_1.sample_posterior(PRNGKey(1), p4_1, (1000,))
print_summary(samples, 0.89, False)
```

| |mean|std|median|5.5%|94.5%|n_eff|r_hat|
|-|-|-|-|-|-|-|-|
|mu   |154.60|0.40|154.60|154.00|155.28| 995.06|1.00|
|sigma|  7.76|0.30|7.76  |7.33  |8.26  |1007.15|1.00|

Các con số này ước lượng kiểu Gaussian cho phân phối biên của mỗi parameters. Có nghĩa là khả năng của mỗi giá trị của $\mu$, sau khi trung bình hoá mọi khả năng của $\sigma$, cho ra phân phối Gaussian với trung bình là 154.6 và độ lệch chuẩn 0.4.

Phân vị 5.5% và 94.5% là ranh giới percentile, tương ứng với khoảng tin cậy 89% (CI 89%). So sánh với kết quả HPDI từ grid approx, ta thấy nó gần như giống nhau. Khi posterior là gần giống Gaussian, đây là điều hiển nhiên bạn sẽ thấy.

Prior ở trên rất yếu, bởi vì nó gần như phẳng và bởi vì có nhiều data. Nếu tôi thêm nhiều thông tin cho prior hơn, bạn sẽ thấy sự ảnh hưởng. Chỉ cần thay độ lệch chuẩn của $\mu$ thành 0.1, thì nó là một prior hẹp.

```python
def model(height):
    mu = numpyro.sample("mu", dist.Normal(178, 0.1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

m4_2 = AutoLaplaceApproximation(model)
svi = SVI(model, m4_2, optim.Adam(1), AutoContinuousELBO(),
          height=d2.height.values)
init_state = svi.init(PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, np.zeros(2000))
p4_2 = svi.get_params(state)
samples = m4_2.sample_posterior(PRNGKey(1), p4_2, (1000,))
print_summary(samples, 0.89, False)
```

| |mean|std|median|5.5%|94.5%|n_eff|r_hat|
|-|-|-|-|-|-|-|-|
|mu   |177.86|0.10|177.86|177.72|178.03| 995.05|1.00|
|sigma| 24.57|0.94| 24.60| 23.01| 25.96|1012.88|1.00|

Chú ý rằng ước lượng của $\mu$ di dời rất ít từ prior. Prior tập trung quanh 178. Nên chuyện này không có gì bất ngờ. Nhưng để ý rằng ước lượng của $\sigma$ tăng rất nhiều, mặc dù ta không đổi gì cả. Khi mà golem khẳng định trung bình ở 178 - như prior - golem phải bù trừ lại bằng $\sigma$, kết quả là một param khác đã bị thay đổi.

### 4.3.6 Lấy mẫu từ quad approx

Phần trên giới thiệu cách tạo ước lượng quad approx cho posterior. Và việc lấy mẫu cũng rất đơn giản, bởi ta cần phải nhận ra posterior này là phân phối Gaussian đa chiều.

Thêm vào đó, ta có thể tính covariance cũng như correlation matrix cho 2 parameter này.

```python
samples = m4_1.sample_posterior(PRNGKey(1), p4_1, (1000,))
vcov = np.cov(np.array(list(samples.values())))
# DeviceArray([[0.16249639, 0.0016826 ],
#             [0.0016826 , 0.08733711]], dtype=float32)
```

Đây là matrix **VARIANCE-COVARIANCE**, nói cho ta biết các parameter liên quan với nhau như thế nào. Nó có thể tạo ra 2 thành phần: (1) variance của mỗi parameter (2) matrix tương quan correlation giữa các parameter.

```python
print(np.diagonal(vcov))
print(vcov / np.sqrt(np.outer(np.diagonal(vcov), np.diagonal(vcov))))
# [0.16251372 0.08735678]
# [[1.         0.01451718]
# [0.01451718 1.        ]]
```

Kết quả trên cùng là variance của $\mu$ và $\sigma$. Kết quả dưới là tương quan giữa $\mu$ và $\sigma$, có giá trị từ -1 đến +1. Giá trị 1 nghĩa là param tương quan với chính nó. Những giá trị khác gần bằng 0 trong ví dụ này. Điều này nói cho ta biết $\mu$ mới không liên quan gì với $\sigma$ và tương tự với $\sigma$.

Làm sao để lấy mẫu từ posterior đa chiều? Thay vì lấy 1 mẫu từ một phân phối Gaussian đơn giản, ta lấy mẫu từ phân phối Gaussian đa chiều.

```python
post = m4_1.sample_posterior(PRNGKey(1), p4_1, (int(1e4),))
pd.DataFrame(post)[:6]
# {'mu': [154.24428, 154.48541, 154.97919, 154.2124, 155.49146, 154.82701],
# 'sigma': [7.560233, 7.3066654, 7.280367, 7.811781, 7.905633, 7.978665]}
```

Kết quả là DataFrame với 2 cột và 1000 dòng, một cột là $\mu$ và cột kìa là $\sigma$. Kết quả này gần giống với MAP phần trước

```python
print_summary(post, 0.89, False)
```

| |mean|std|median|5.5%|94.5%|n_eff|r_hat|
|-|-|-|-|-|-|-|-|
|mu|   154.61|0.41|154.61|153.93|155.25|9927.00| 1.00|
|sigma|  7.75|0.29|  7.74|  7.28|  8.22|9502.46| 1.00|

## <center>4.4 Linear model</center><a name="4"></a>

Chúng ta đã làm model Gaussian về chiều cao của người lớn trong quần thể. Nhưng nó không có cảm giác "regression" nào cả. Chúng ta ở đây cần model tạo kết quả liên quan đến variable khác, **PREDICTOR VARIABLE**. Nếu pred var có liên quan thống kê đến outcome, ta có thể dùng nó để dự đoán outcome. Và nó là linear regression.

Ta sẽ dùng data trên, height (outcome) sẽ thay đổi như thế nào với weight (pred var). Trước tiên ta vẽ scatter thể hiện mối liên quan giữa height và weight trong data.

```python
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"));
```

![](/assets/images/figure 4-11.svg)

Bạn thấy rõ có quan hệ giữa height và weight. Biết được weight sẽ đoán được height của người đó.

### 4.4.1 Linear model

Mục tiêu ở đây là tạo parameter cho mean của phân phối Gaussian, $\mu$, thành hàm linear của pred var và những parameter khác mà mình tự chế. Cách định nghĩa này giúp golem hiểu rằng pred var có quan hệ hằng định và tính cộng thêm cho mean của outcome. Sau đó golem tính phân phối posterior với quan hệ này.

Có nghĩa là, cỗ máy xem xét mọi khả năng có thể của tập hợp parameter. Với model linear, vài parameter sẽ là độ mạnh của quan hệ giữa trung bình outcome, $\mu$, và giá trị của variable khác. Với mỗi kết hợp của giá trị, cỗ máy tính xác suất posterior. Phân phối posterior xếp hạng vô số khả năng kết hợp của giá trị parameter bằng logic. Kết quả là, phân phối posterior cung cấp khả năng tương đối của các quan hệ khác nhau.

Model được định nghĩa như sau: Giả sử x là cột weight của data, $\bar{x}$ là trung bình của x.

$$ \begin{matrix}
h_i & \sim \text{Normal}(\mu_i, \sigma) \quad && [\text{likelihood}]\\
\mu_i & =\alpha + \beta (x_i - \bar{x}) \quad && [\text{linear model}]\\
\alpha & \sim \text{Normal}(178, 20) \quad && [\alpha \text{prior}]\\
\beta & \sim \text{Normal} (0, 10) \quad && [\beta \text{prior}]\\
\sigma & \sim \text{Uniform} (0, 50) \quad && [\sigma \text{prior}]
\end{matrix}$$

Ta sẽ thảo luận từng dòng một.

#### 4.4.1.1 Xác suất của data.

Hãy bắt đầu bằng likelihood, nó không khác gì model cũ, ngoại trừ có thêm index *i* ở $\mu$. Ta có thể đọc $h_i$ là mỗi *h*, $\mu_i$ là mỗi $\mu$. Trung bình $\mu$ bây giờ dựa vào giá trị cụ thể ở hàng *i*. Cho nên ký hiệu *i* của $\mu$ nghĩa là *trung bình dựa vào hàng*.

#### 4.4.1.2 Linear model

Trung bình $\mu$ không còn là parameter ta cần ước lượng, mà nó được tạo ra từ những parameter khác, $\alpha$ và $\beta$. Nó không phải quan hệ stochastic (ngẫu nhiên phân phối) - không phải dấu $\sim$ - bởi vì định nghĩa của $\mu_i$ là mang tính quyết định. Có nghĩa là, khi ta biết $\alpha$ và $\beta$ và $x_i$, ta biết $\mu_i$.

Giá trị $x_i$ chỉ là giá trị weight ở hàng *i*. Và nó cũng dẫn tới cùng cá thể có chiều cao $h_i$ cùng hàng. $\alpha$ và $\beta$ thì bí ẩn hơn. Chúng từ đâu? Tôi tự tạo chúng. Parameter $\mu$ và $\sigma$ đủ để mô tả một phân phối Gaussian. Còn $\alpha$ và $\beta$ là những thiết bị ta tạo ra để xử lý $\mu$, cho phép nó biến thiên có hệ thống xuyên suốt data.

Bạn sẽ tạo được nhiều dạng parameter khi kỹ năng bạn khá hơn. Một hướng suy nghĩ là những parameter tự tạo này là mục tiêu để cỗ máy học. Mỗi parameter là thứ gì đó phải được mô tả ở posterior. Nên khi bạn muốn biết gì đó về data, bạn hỏi golem bằng cách tạo mới parameter cho nó. Từ từ bạn sẽ hiểu khi bạn tiếp tục học. 

Dòng 2 nói cho regression golem rằng bạn có 2 câu hỏi về trung bình của outcome.
1. Chiều cao mong đợi khi $x_i = \bar{x}$? Parameter $\alpha$ sẽ trả lời câu hỏi này, bởi vì khi $x_i = \bar{x}$, $\mu_i = \alpha$. Với lý do này, $\alpha$ thường được gọi là *intercept*. Nhưng ta không nên nghĩ nó là gì đó trừu tượng, mà nên nghĩ nó là ý nghĩa với khía cạnh của data.
2. Thay đổi height như thế nào khi $x_i$ tăng 1 đơn vị? Param $\beta$ trả lời câu hỏi này. Nó thường được gọi là *slope*. Ta nên nghĩ nó là tốc độ thay đổi của giá trị mong đợi.

Hợp lại, 2 param này hỏi golem tìm một đường thẳng để liên kết *x* tới *h*, đường thẳng này đi qua $\alpha$ khi $x_i = \bar{x}$ và có *slope* $\beta$.

>**Nghĩ lại: Không có gì đặc biệt hay tự nhiên về linear model.** Bạn có thể chọn mối quan hệ giữa $\alpha$ và $\beta$ và $\mu$. Ví dụ, model dưới đây hoàn toàn hợp lý:  
$$ \mu_i = \alpha \;exp(-\beta x_i)$$  
Này không phải linear regression, nhưng nó định nghĩa một regression model. Quan hệ linear mà ta dùng là để cho thuận tiện, nhưng không có gì bắt ta phải dùng nó. Nó rất phổ biến ở vài lĩnh vực, như kinh tế hay dân số, dùng dạng hàm số của $\mu$ từ giả thuyết hơn lài tính địa tâm của linear model. Model dựng trên giả thuyết có thể hoạt động tốt hơn nhiều so với linear model trên cùng một hiện tượng.

---

**Nghĩ nhiều hơn: Đơn vị.** Ta có thể dựng lại model có đơn vị cho rõ ràng hơn.

$$ \begin{matrix}
h_i \; cm & \sim \text{Normal}(\mu_i\; cm, \sigma\; cm)\\
\mu_i\; cm & =\alpha \; cm+ \beta \frac{cm}{kg}(x_i \;kg - \bar{x}\; kg)\\
\end{matrix}$$

Bằng cách thêm đơn vị cho công thức, ta thấy rõ $\beta$ là tốc độ thay đổi cm trên kg. Có một truyền thống là *phân tích vô chiều* rằng không cần dùng đơn vị. Ví dụ, ta có thể dùng height chia cho một đơn vị nào đó, loại bỏ đơn vị của nó. Đơn vị đo lường là một loại hướng dẫn tuỳ ý của loà người, đôi khi phân tích không có đơn vị thì tự nhiên và tổng quát hơn.

---

#### 4.4.1.3 Priors 

Các dòng còn lại mô tả phân phối prior của parameter. Có 3 param là $\alpha, \beta, \sigma$. Bạn đã gặp prior $\alpha$ và $\sigma$, nhưng $\alpha$ phần trước gọi là $\mu$.

Prior $\beta$ có lẽ cần giải thích. Tại sao có Gaussian prior với trung bình là zero? Prior này có xác suất >0 và <0 là bằng nhau, khi $\beta = 0$ thì weight không liên quan với height. Để dánh giá prior này tạo ra gì, ta sẽ mô phỏng phân phối prior prediction.

Mục tiêu là mô phỏng height từ model bằng prior. Trước tiên, ta xem xét  khoảng giá trị weight mô phỏng. Có thể dùng khoảng giá trị weight quan sát được. Ta sẽ vẽ các đường thẳng, dùng 100 giá trị từ prior $\alpha$ và $\beta$.

```python
with numpyro.handlers.seed(rng=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 20), sample_shape=(N,))
    b = numpyro.sample("b", dist.Normal(0, 10), sample_shape=(N,))
plt.subplot(xlim=(d2.weight.min(), d2.weight.max()), ylim=(-100, 400),
            xlabel="weight", ylabel="height")
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("b ~ Normal(0, 10)")
xbar = d2.weight.mean()
x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
for i in range(N):
    plt.plot(x, a[i] + b[i] * (x - xbar), "k", alpha=0.2);
```

![](/assets/images/figure 4-12.svg)

Để so sánh, ta có thêm đường zero và đường 272 là chiều cao của người cao nhất thế giới. Kết quả này không giống quần thể loài người. Nó nói rằng mối quan hệ giữa weight và height có thể âm và dương rất lớn. Trước khi nhìn thấy data, đây là model rất kém.

Ta biết rằng chiều cao trung bình tăng theo cân nặng trung bình, ít ra dưới một mức nào đó. Hãy thử giới hạn nó bằng giá trị dương. Cách đơn giản nhất là định nghĩa prior bằng Log-Normal. Log-Normal (0, 1) sau khi logarith sẽ thành normal (0, 1).

$$ \beta \sim \text{Log-Normal}(0, 1)$$

```python
b = dist.LogNormal(0, 1).sample(PRNGKey(0), (int(1e4),))
az.plot_kde(b);
```

![](/assets/images/figure 4-13.svg)

Nếu log của $\beta$ là normal, thì bản thân $\beta$ phải dương. Lý do là exp(x) luôn lớn hơn zero với bất kỳ số thực x. Đó là lý do tại sao Log-Normal khá phổ biến để ép buộc quan hệ dương. Giờ mô phỏng lại với prior $\beta$ mới.

![](/assets/images/figure 4-14.svg)

Biểu đồ này có vẻ hợp lý hơn. Vẫn có vài quan hệ phi lý. Nhưng đa số nằm trong khoảng bình thường của con người.

Ta đang tuỳ biến prior, nhưng về sau bạn sẽ thấy khi có quá nhiều data thì việc chọn prior không còn ý nghĩa nữa. Có 2 lý do ta tuỳ biến. Một, có quá nhiều phân tích mà không kích cỡ data nào làm cho prior không liên quan. Trong trường hợp đó, quy trình non-Bayes cũng không làm được gì hơn. Nó dựa rất nhiều vào cấu trúc của model. Chú ý đến việc chọn prior lúc ấy là rất cần thiết. Thứ hai, nghĩ về prior giúp ta phát triển model tốt hơn, thậm chí hơn cả model địa tâm.

>**Nghĩ lại: Prior đúng.** Nhiều người hỏi rằng prior đúng cho một phân tích nào đó là gì. Câu hỏi này suy ra với một nghiên cứu nào đó, ta cần phải chọn prior đúng, nếu không nghiên cứu sẽ sai. Điều này không đúng. Có rất nhiều prior đúng nhưng likelihood phải chính xác. Model thống kê là cỗ máy cho suy luận.  
Có một guideline hướng dẫn bạn chọn prior. Prior mã hoá tình trạng thông tin trước khi thấy data. Cho nên prior cho phép chúng ta khám phá các hệ quả khởi đầu bằng thông tin khác nhau. Trong trường hợp ta có thông tin prior tốt về khả năng của param, như quan hệ âm giữa height và weight, ta có thể mã hoá trực tiếp vào prior. Khi không có đủ thông tin, ta vẫn biết được khoảng giá trị của nó. Bạn có thể thay đổi prior và lặp lại phân tích để xem thử với tình trạng thông tin ban đầu khác nhau thì ảnh hưởng suy luận như thế nào. Thông thường, có rất nhiều lựa chọn prior, và chúng đều cho suy luận như nhau. Và Prior Bayesian thuận tiện thì tương đương với cách tiếp cận non-Bayes.  
Sự lựa chọn hay làm cho người mới lo lắng. Đây là một ảo giác mà những quy trình có hướng đối tương hơn những quy trình cần lựa chọn của người dùng, như chọn prior. Nếu đúng, thì mọi người đều là như nhau. Nó không mang ý nghĩa thực tế hay chính xác.

>**Nghĩ lại: mô phỏng prior predictive và p-hacking.** Một vấn nạn trong thống kê ứng dụng là "p-hacking", một hành vi chỉnh sửa model và data để có được kết quả mong muốn, thường là nhỏ < 5%. Vấn đề là model bị chỉnh sửa sau khi thấy data, p-value không còn ý nghĩa gốc của nó. Kết quả sai là hiển nhiên. Chúng ta không quan tâm đến p-value trong sách này. Nhưng nguy hiểm vẫn còn đó, nếu chúng ta chọn prior dựa vào quan sát, để có được kết quả mong muốn. Quy trình ta vừa làm là chọn prior trước khi có data, những giới hạn, khoảng giá trị, quan hệ lý thuyết. Đó là lý do data chưa xuất hiện ở phần trước. Ta chọn prior dựa vào sự thật, không phải từ data.

### 4.4.2 Tìm phân phối posterior

Ta xem lại định nghĩa model:

$$ \begin{matrix}
h_i & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i & =\alpha + \beta (x_i - \bar{x}) \\
\alpha & \sim \text{Normal}(178, 20) \\
\beta & \sim \text{Log-Normal} (0, 10) \\
\sigma & \sim \text{Uniform} (0, 50)
\end{matrix}$$

Ta dùng quad approx để fit model

```python
# load data again, since it's a long way back
Howell1 = pd.read_csv("../data/Howell1.csv", sep=";")
d = Howell1
d2 = d[d["age"] >= 18]

# define the average weight, x-bar
xbar = d2.weight.mean()

# fit model
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + b * (weight - xbar)
    numpyro.sample("mu", dist.Delta(mu), obs=mu)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

m4_3 = AutoLaplaceApproximation(model)
svi = SVI(model, m4_3, optim.Adam(1), AutoContinuousELBO(),
          weight=d2.weight.values, height=d2.height.values)
init_state = svi.init(PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, np.zeros(1000))
p4_3 = svi.get_params(state)
```

>**Nghĩ lại: Mọi thứ có dựa trên parameter đều có phân phối posterior.** Trong model ở trên, parameter $\mu$ không còn là parameter, vì nó trở thành hàm số của parameter $\alpha$ và $\beta$. Nhưng vì $ parameter $\alpha$ và $\beta$ có joint posterior, và $\mu$ cũng vậy. Phần sau, ta làm việc trực tiếp với phân phối posterior của $\mu$, mặc dù nó không còn parameter.  Bởi vì parameter là bất định, mọi thứ dựa vào nó đều bất định. Bao gồm số thống kê như $\mu$, cũng như dự đoán của model, đo đạc mức độ fit, và mọi thứ khác dựa trên parameter. Bằng cách lấy mẫu từ posterior, việc bạn cần làm là đưa tính bất định vào đại lượng cần dùng.

---

**Nghĩ nhiều hơn: log và exp.** Rất nhiều nhà khoa học tự nhiên và xã hội quên logarithm. Logarithm xuất hiện rất nhiều trong thống kê ứng dụng. Bạn có thể nghĩ y=log(x) là gán y cho số mũ của x. Hàm x=exp(y) thì ngược lại. Định nghĩa này khó hiểu. Nhưng nó rất nhiều phép tính máy tính dựa vào nó.  
Định nghĩa này cho phép code Log-Normal prior cho $\beta$ bằng cách khác.  
```python
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    log_b = numpyro.sample("log_b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + np.exp(log_b) * (weight - xbar)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
```

---

### 4.4.3 Diễn giải phân phối posterior

Vấn đề của model thống kê là nó khó hiểu. Sau khi fit model, nó chỉ báo cáo lại phân phối posterior. Kết quả đó là đúng. Nhưng nghĩa vụ của bạn là xử lý và hiểu nó.

Có 2 cách xử lý chính: (1) đọc bảng (2) vẽ đồ hoạ mô phỏng. Với câu hỏi đơn giản, ta có thể học rất nhiều từ bảng giá trị biên. Nhưng phần khó khăn nhất của bảng là sự quá đơn giản trong khi model phức tạp hơn nhiều. Tác giả sẽ nhấn mạnh vẽ đồ hoạ phân phối posterior và posterior prediction.

#### 4.4.3.1 Bảng giá trị biên.

Sau khi fit data vào linear model, ta có thể kiểm tra bảng giá trị biên của từng parameter.

```python
samples = m4_3.sample_posterior(PRNGKey(1), p4_3, (1000,))
print_summary(samples, 0.89, False)
```

| |mean|std|median|5.5%|94.5%|n_eff|r_hat|
|-|-|-|-|-|-|-|-|
|a    |154.62|0.27|154.63|154.16|155.03| 931.50|1.00|
|b    |  0.91|0.04|  0.90|  0.84|  0.97|1083.74|1.00|
|sigma|  5.08|0.19|  5.08|  4.79|  5.41| 949.65|1.00|

Ta sẽ tập trung vào $\beta$, bởi vì nó là *slope*, giá trị 0.91 có thể đọc là một người hơn 1 kg cân nặng thì cao thêm 0.91 cm. 89% của posterior nằm ở 0.84, 0.97. Điều này có nghĩa $\beta$ gần zero hay hơn 1 đều không phù hợp với data và model. Đây rõ rằng không phải bằng chứng rằng mối quan hệ giữa weight và height là linear, bởi vì model chỉ nhận đường thẳng. Nó chỉ nói rằng, nếu bạn chọn đường thẳng, thì đường thẳng 0.9 là phù hợp nhất.

Nhớ rằng, con số ở bảng trên không đủ để mô tả quad approx posterior. Ta cần thêm variance-covariance matrix.

```python
vcov = np.cov(np.array(list(post.values())))
np.round(vcov, 3)
```

| |a| b| sigma|
|-|-|-|-|
| a| 0.073| 0.000| 0.000| 
|b| 0.000| 0.002| 0.000|
| sigma| 0.000| 0.000| 0.037|

Có rất ít covariance giữa các parameters. Nguyên nhân có thể do **CENTERING**.

#### 4.4.3.2 Vẽ posterior so với data

Plot posterior với data luôn luôn có ích. Nó không chỉ giúp ta diễn giải posterior, mà còn kiểm tra giả định của model. Nếu như suy luận của model khác xa với mẫu quan sát được, có thể model fit chưa tốt hoặc model thiết kế sai. Nhưng ngay cả khi bạn chỉ dùng plot để kiểm tra posterior, nó là một công cụ quý báu. Với model đơn giản như này, ta có thể đọc kết quả từ bảng, nhưng với model phức tạp hơn, đặc biệt là model chứa hiệu ứng tương tác (interaction), diễn giải posterior rất khó. Cùng với bảng covariance, plot là không thể thiếu được.

Ta sẽ làm một phiên bản đơn giản của công việc, chỉ ghép giá trị trung bình của posterior vào data. Và sau đó sẽ thêm nhiều thông tin hơn.

Trước mắt, ta sẽ vẽ data và một đường thẳng là trung bình của a và b.

```python
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"))
post = m4_3.sample_posterior(PRNGKey(1), p4_3, (1000,))
a_map = np.mean(post["a"])
b_map = np.mean(post["b"])
x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
plt.plot(x, a_map + b_map * (x - xbar), "k");
```

![](/assets/images/figure 4-15.svg)

#### 4.4.3.3 Thêm tính bất định quanh trung bình

Đường thẳng trung bình posterior ở trên là đường thẳng có khả năng cao nhất trong vô số đường thẳng có thể của trung bình posterior. Plot như trên rất có ích, nó tạo ấn tượng cho mức độ ước lượng suy luận. Nhưng nó không cho thấy tính bất định. Nhớ rằng, phân phối posterior xem xét mọi đường thẳng regression giữa weight và height. Mỗi cặp $\alpha$ và $\beta$ đều có phân phối posterior. Có thể có rất nhiều đường thẳng trung bình rải đều, có cùng khả năng so với đường thẳng trên. Hoặc chúng tập trung quanh đường thẳng trên.

Để thêm tính bất định, ta lấy mẫu $\alpha$ và $\beta$ để vẽ đường thẳng. Các bạn sẽ thấy cảm nhận tốt hơn nếu ta bắt đầu với ít data hơn. Trước mắt là 10 data. Sau đó tăng dần lên 50, 150, 352. Với mỗi cỡ data, ta lấy mẫu 20 cặp $\alpha$ và $\beta$.

```python
def model(weight, height):
        a = numpyro.sample("a", dist.Normal(178, 20))
        b = numpyro.sample("b", dist.LogNormal(0, 1))
        sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
        mu = a + b * (weight - np.mean(weight))
        numpyro.sample("mu", dist.Delta(mu), obs=mu)
        numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

fig, ax = plt.subplots(2,2, figsize=(10,8))

from itertools import product

for (row, col), n in zip(product([0,1], [0,1]), [10,50,150,352]):
    data = d2[:n]
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide, optim.Adam(1), AutoContinuousELBO(),
              weight=data.weight.values, height=data.height.values)
    state, loss = lax.scan(lambda s, l: svi.update(s), svi.init(PRNGKey(0)), np.zeros(1000))
    params = svi.get_params(state)
    post = guide.sample_posterior(PRNGKey(1), params, (20,))
    # display raw data and sample size
    az.plot_pair(data[["weight", "height"]].to_dict("list"), ax=ax[row][col])
    ax[row][col].set(
        xlim=(d2.weight.min(), d2.weight.max()),
        ylim=(d2.height.min(), d2.height.max()),
        title="N = {}".format(n)
    )
    # plot the lines, with transparency
    x = np.linspace(d2.weight.min(), d2.weight.max(), 20)
    for i in range(20):
        ax[row][col].plot(x, post["a"][i] + post["b"][i] * (x - dN.weight.mean()),
                     "k", alpha=0.3)
plt.tight_layout()
```

![](/assets/images/figure 4-16.svg)

Hình vẽ cho khía cạnh tin cậy cao và tin cậy thấp của data. Đám mây các đường regression cho thấy mức độ bất định cao hơn ở 2 cực của weight.

Khi số data tăng lên, đám mây này tập trung lại hơn ở đường trung bình. Này nghĩa là model càng tin cậy hơn với vị trí của trung bình.

#### 4.4.3.4 Vẽ khoảng tin cậy của trung bình

Bốn hình trên nhìn rất đẹp, vì nó thể hiện tính bất định của quan hệ mà mọi người đều cảm thấy dễ hiểu. Nhưng có phương pháp phổ biến và trực quan hơn, đó là vẽ khoảng tin cậy xung quanh đường trung bình. 

Thử với điểm weight = 50 kg, ta tạo 10,000 giá trị của $\mu$ cho cá nhân có 50 kg, bằng phương pháp lấy mẫu:

```python
posterior_sample = guide.sample_posterior(PRNGKey(11), params, (int(1e5),))
posterior_predictor = Predictive(m4_3.model, posterior_sample)
mu_posterior_at50 = posterior_predictor(PRNGKey(2), 50, None)['mu']
az.plot_dist(mu_posterior_at50, bw=1, label="mu|weight=50")
```

![](/assets/images/figure 4-17.svg)

Bởi vì các thành phần của $\mu$ có phân phối, nên $\mu$ cũng vậy. Nhưng bởi vì $\alpha$ và $\beta$ là Gaussian, nên phân phối của $\mu$ cũng là Gaussian.

Vì nó là phân phối nên ta có tìm khoảng tin cậy cho nó. CI 89% với $\mu =50$:

```python
np.percentile(mu_at_50, q=(5.5, 94.5))
# DeviceArray([158.5957 , 159.71445], dtype=float32)
``` 

Nghĩ là khoảng 89% ở trung tâm phân phối là 159cm và 160cm, giả định vào data, model, weight=50.

Điều ta cần làm là lặp lại phép tính trên với mọi giá trị weight ở trục hoành, không chỉ là 50 kg. Và lấy CI 89% tại mỗi điểm. Ta có thể dùng các điểm weight của data, hoặc khoảng giá trị weight cách đều trong khoảng min và max của data.

```python
weight_seq = np.arange(start=25, stop=71, step=1)
posterior_predictor = Predictive(m4_3.model, posterior_sample)
mu_samples = posterior_predictor(PRNGKey(2), weight_seq, None)["mu"]
mu_samples.shape
# (100000, 46)
```

Ta có thể tìm trung bình và khoảng CI 89% tại mỗi điểm trong `weight_seq`:

```python
mu_mean = np.mean(mu, 0)
mu_PI = np.percentile(mu, q=(5.5, 94.5), axis=0)
```

Vẽ đồ hoạ biểu diễn các mẫu $\mu$ tại mỗi điểm và khoảng 89% của đường trung bình.

```python
fig, ax = plt.subplots(1, 2, figsize=(10,5))
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"),
             scatter_kwargs={"alpha": 0.8}, ax=ax[1])
for i in range(30):
    ax[0].plot(weight_seq, mu_samples[i], "o", c="royalblue", alpha=0.1)
ax[1].plot(weight_seq, mu_mean, "k")
ax[1].fill_between(weight_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2);
```

![](/assets/images/figure 4-18.svg)

Mặc dù phương pháp dùng công thức toán học dễ dàng hơn trong việc tính khoảng tin cậy như thế này. Nhưng phương pháp lấy mẫu linh hoạt hơn và cho phép đông đảo khán giả hơn để lấy insight từ model thống kê. Một lần nữa, với phương pháp ước lượng từ MCMC, đây là cách duy nhất khả thi. Nên phải học nó từ bây giờ.

>**Nghĩ lại: Cẩn thận khi dùng Khoảng tin cậy.** Khoảng tin cậy ở hình trên bám chắc xung quanh đường MAP. Nghĩa là có rất ít tính bất định quanh height hay kết quả của hàm số của weight. Tuy nhiên model sai, kém cũng có thể có CI hẹp. Hãy nhớ rằng suy luận dựa vào model được thiết kế. Hình trên cũng có thể diễn giải rằng: Điều kiện là giả định height và weight liên quan với nhau bằng quan hệ linear, thì đường thẳng khả dĩ nhất là đường thẳng đó cùng với khoảng tin cậy của nó.

#### 4.4.3.5 Khoảng dự báo

Nãy giờ ta làm khoảng 89% của $\mu$, chứ không phải là height thực tế. Nhìn lại model, ta thấy height phân phối theo Gaussian($\mu, \sigma$). Ta chỉ lấy mẫu tính bất định của $\mu$, nhưng thực tế dòng trên nói rằng height quan sát được trải đều quanh $\mu$ với độ lệch chuẩn $\sigma$. Ta phải lồng ghép $\sigma$ bằng cách nào đó.

Cách làm như sau: Đầu tiên ta thực hiện các quy trình như đã nói trên, lấy mẫu (ở đây là 1000) 1000 mẫu a, b, sigma posterior. Từ 1000 mẫu a,b ta tìm $\mu$ tại mỗi điểm giá trị của weight. Trong data có 46 giá trị weight từ min đến max. Từ 1000 điểm $mu$ tại mỗi giá trị weight đó, ta vẽ được đường thẳng trung bình và khoảng tin cậy 89% của nó. Sau đó, ta từ $\mu$ posterior và $\sigma$ posterior, tạo Normal distribution, lấy 1 mẫu từ phân phối đó. Từ kết quả này, ta vẽ được khoảng tin cậy của dự báo.

```python
sim_height_predictor = Predictive(m4_3.model, post, return_sites=["height"])
sim_height = sim_height_predictor(PRNGKey(2), weight_seq, None)["height"]
sim_height.shape # (1000, 46)

height_PI = np.percentile(sim_height, q=(5.5, 94.5), axis=0)

az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"),
             plot_kwargs={"alpha": 0.5})

# draw MAP line
plt.plot(weight_seq, mu_mean, "k")

# draw HPDI region for line
plt.fill_between(weight_seq, mu_PI[0], mu_HPDI[1], color="k", alpha=0.2)

# draw PI region for simulated heights
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k",
                 alpha=0.15);
```

![](/assets/images/figure 4-19.svg)

Chú ý rằng đường viền của khoảng tin cậy dự báo hơi gồ ghề. Đó là do biến thiên của kết quả mô phỏng ở hai đuôi của Gaussian. Nếu thấy khó chịu, bạn có thể nâng tổng số mẫu lên. Rất may là chuyện ấy không quan trọng, ngoài tính thẩm mỹ. Hơn nữa, nó nhắc chúng ta rằng mọi suy luận thống kê đều mang tính chất ước lượng.

>**Nghĩ lại: 2 loại bất định.** Trong quy trình trên, ta gặp tính bất định của parameter và tính bất định trong việc lấy mẫu. Đây là 2 khái niệm khác nhau, mặc dù cách làm như nhau và trộn chúng trong mô phỏng posterior predictive. Phân phối posterior là xếp hạng của khả năng tương đối của mọi sự kết hợp của giá trị parameter. Sự khác biệt của kết quả mô phỏng, như height, là phân phối gồm biến thiên lấy mẫu từ quy trình Gaussian. Và nó cũng là một giả định model. Cả hai tính bất định này đều quan trọng

## <center>4.5 Cong từ thẳng</center><a name="5"></a>

Trước khi gặp model dùng nhiều predictor hơn ở chương sau, ta hãy làm quen những hàm số làm cong. Model phần trước giả định có mối quan hệ linear giữa predictor và outcome. Không có gì đặc biệt về nó, ngoài sự dễ dàng.

Có 2 phương pháp dùng linear regression để tạo đường cong. Một là **POLYNOMIAL REGRESSION** và hai là **B-SPLINES**. Cả hai phương pháp đều biến đổi chỉ một predictor thành nhiều variable tự chế. Nhưng spline thì có nhiều ưu thế hơn. Cả hai phương pháp đều mô tả hàm số liên quan một variable đến outcome. Causal inference, suy luận nhân quả, đòi hỏi hơn so với linear regression.

### 4.5.1 Polynomial Regression

Polynomial (đa bậc) dùng luỹ thừa của variable - bình phương hoặc lập phương - làm biến predictor thêm. Đây là phương pháp tạo đường cong đơn giản. Nó rất phổ biến, và hiểu cách nó thực hiện giúp bạn hiểu hơn các model về sau. Để bắt đầu, ta dùng lại data `Howell1` cũ. Nếu nhìn vào scatter_plot, ta thấy mỗi quan hệ giữa `weight` và `height` hơi cong, bởi vì ta thêm những cá thể trẻ em.

Polynomial regression thông dụng nhất là model parabol của trung bình. Gọi x là weight đã được chuẩn hoá. Thì phương trình bậc 2 của trung bình chiều cao có dạng:

$$ \mu_i = \alpha + \beta_1x_i + \beta_2x_i^2 $$

Phần $\alpha + \beta_1x_i$ cũng giống như hàm linear trong linear regressino. Phần thêm vào dùng bình phường $x_i$ để tạo parabol, hơn là đường thẳng tuyệt đối.

Fit data và model cũng dễ. Diễn giải thì khó hơn. Trước tiên, ta phải **STANDARDIZE** predictor. Nó rất có ích trong polynomial. Vì khi có số rất lớn trong data, phần mềm có thể sai, ngay cả phần mềm tốt nhất. Định nghĩa model như sau:

$$ \begin{matrix}
h_i & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i & =\alpha + \beta_1 x_i + \beta_2 x_i^2 \\
\alpha & \sim \text{Normal}(178, 20) \\
\beta_1 & \sim \text{Log-Normal} (0, 1) \\
\beta_2 & \sim \text{Normal} (0, 1) \\
\sigma & \sim \text{Uniform} (0, 50)
\end{matrix}$$

Cái khó ở đây là gán prior cho $\beta_2$, parameter cho giá trị bình phương của x. Không giống như $\beta_1$, ta không cần số dương. Đa số các parameter của bậc cao rất khó hiểu, nhưng mô phỏng prior predict sẽ giúp ta chuyện đó. Ta sẽ dùng quad approx để fit data vào model.

```python
d = pd.read_csv("../data/Howell1.csv", sep=";")
d["weight_s"] = (d.weight - d.weight.mean()) / d.weight.std()
d["weight_s2"] = d.weight_s ** 2

def model(weight_s, weight_s2, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + b1 * weight_s + b2 * weight_s2
    numpyro.sample("mu", dist.Delta(mu), obs=mu)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

m4_5 = AutoLaplaceApproximation(model)
svi = SVI(model, m4_5, optim.Adam(0.3), AutoContinuousELBO(),
          weight_s=d.weight_s.values, weight_s2=d.weight_s2.values,
          height=d.height.values)
init_state = svi.init(PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, np.zeros(1000))
p4_5 = svi.get_params(state)
```

Bảng giá trị biến của các parameter đôi khi rất khó diễn giải.

```python
samples = m4_5.sample_posterior(PRNGKey(1), p4_5, (1000,))
print_summary(samples, 0.89, False)
```

| |mean|std|median|5.5%|94.5%|n_eff|r_hat|
|-|-|-|-|-|-|-|-|
a    |146.05|0.36|146.03|145.47|146.58|1049.96|1.00|
b1   | 21.75|0.30| 21.75| 21.25| 22.18| 886.88|1.00|
b2   | -7.79|0.28| -7.79| -8.21| -7.32|1083.62|1.00|
sigma|  5.78|0.17|  5.78|  5.49|  6.02| 973.22|1.00|

$\alpha$ vần là *intercept*, nó nói giá trị `height` khi `weight` ở trung bình. Nhưng nó không phải giá trị trung bình của `height` trong mẫu. Ta cần phải plot hình lên để xem các param nói gì. Ta sẽ làm tương tự như linear regression

```python
weight_seq = np.linspace(start=-2.2, stop=2, num=30)
pred_dat = {"weight_s": weight_seq, "weight_s2": weight_seq ** 2,
            "height": None}
post = m4_5.sample_posterior(PRNGKey(1), p4_5, (1000,))
predictive = Predictive(m4_5.model, post, return_sites=["mu", "height"])
mu = predictive.get_samples(PRNGKey(2), **pred_dat)["mu"]
mu_mean = np.mean(mu, 0)
mu_PI = np.percentile(mu, q=(5.5, 94.5), axis=0)
sim_height = predictive.get_samples(PRNGKey(2), **pred_dat)["height"]
height_PI = np.percentile(sim_height, q=(5.5, 94.5), axis=0)

az.plot_pair(d[["weight_s", "height"]].to_dict(orient="list"),
             plot_kwargs={"alpha": 0.5})
plt.plot(weight_seq, mu_mean, "k")
plt.fill_between(weight_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k",
                 alpha=0.15);
```

![](/assets/images/figure 4-20.svg)

Mặc dù có vẻ nó fit data tốt hơn, nhưng chưa chắc gì model dễ hiểu. Nó là một mô tả địa tâm của mẫu. Nhưng có hai vấn đề. Một, model fit tốt hơn chưa chắc là model tốt hơn. Hai, nó không mang ý nghĩa sinh học. Ta sẽ gặp lại suy luận nhân quả ở các chương cuối.

>**Nghĩ lại: ý nghĩa của linear.** Model parabole như trên cũng là model "linear" của trung bình, mặc dù phương trình rõ ràng không phải đường thẳng. Tuy nhiên, từ "linear" có thể có nhiều ý nghĩa khác trong ngữ cảnh khác, và nhiều người dùng ý nghĩa khác nhau trong cùng ngữ cảnh. Chữ "linear" trong ngữ cảnh này là $\mu_i$ là hàm linear của bất kỳ parameter nào. Model như vậy có ưu thế là fit dễ hơn. Và nó cũng dễ diễn giả hơn, bởi vì giả định của nó là mọi parameter ảnh hưởng độc lập đến trung bình của outcome. Nó có nhược điểm là bị dùng vô tội vạ. Khi bạn có kiến thức chuyên môn, nó sẽ làm việc tốt hơn là model linear. Model này là những thiết bị địa tâm để mô tả tương quan từng phần. Chúng ta nên cảm thấy xấu hổ khi dùng nó, để ta không trở nên tự mã với cách giải thích mang tính hiện tượng từ kết quả của nó.

---

**Nghĩ lại: Trở về scale cũ.** Plot trên dùng đơn vị đã chuẩn hoá ở trục hoành. Chúng thường gọi là *z-scores*. Giả sử bạn fit model bằng variable đã chuẩn hoá, những muốn plot ở scale cũ, bạn chỉ cần chuyển chúng về như sau:

```python
ticks = np.linspace(d['weight_s'].min(), d['weight_s'].max(), 5)
labels = np.round(ticks * d['weight'].std() + d['weight'].mean(),1)
plt.xticks(ticks=ticks, labels=labels);
```

---

### 4.5.2 Splines

Cách thứ hai để tạo đường cong là xây dựng **SPLINE**. Trong thống kê, spline là một hàm số làm mượt dựa trên nhiều hàm số thành phần. Có rất nhiều loại Spline. **B-SPLINE** là thường gặp nhất. Chữ "B" là "basis", nghĩa là "thành phần". B-spline gộp nhiều hàm số zig zag lại với nhau, gọi là hàm cơ bản. Mặc dù có rất nhiều loại spline, ta học B-spline vì ta cần phải quyết định vài chọn lựa mà những spline khác tự động hoá. Bạn cần phải hiểu B-spline trước khi hiểu những spline phức tạp hơn.

Để bắt đầu, ta cần một data zigzag hơn. Hoa anh đào ở Nhật nở hoa khắp nơi vào mùa xuân hàng năm. Thời điểm để nở hoa dao động rất nhiều tuỳ theo năm và thập kỷ.

```python
d= pd.read_csv("../data/cherry_blossoms.csv", sep=";")
d.describe()
```

|     |     year |       doy |        temp |   temp_upper |   temp_lower |
|:----|---------:|----------:|------------:|-------------:|-------------:|
|count| 1215     | 827       | 1124        |  1124        |   1124       |
|mean | 1408     | 104.541   |    6.14189  |     7.18515  |      5.09894 |
|std  |  350.885 |   6.40704 |    0.663648 |     0.992921 |      0.85035 |
|min  |  801     |  86       |    4.67     |     5.45     |      0.75    |
|25%  | 1104.5   | 100       |    5.7      |     6.48     |      4.61    |
|50%  | 1408     | 105       |    6.1      |     7.04     |      5.145   |
|75%  | 1711.5   | 109       |    6.53     |     7.72     |      5.5425  |
|max  | 2015     | 124       |    8.3      |    12.1      |      7.74    |

Ta chỉ thực hành trên `doy` là ngày đầu tiên nở hoa. Nó có khoảng từ min=86 đến max=124, tức cuối tháng 3 và đầu tháng 5. Năm ghi nhận hoa nở từ 801 đến 2015. Bạn có thể scatterplot ra xem quan hệ giữa `doy` và `year`, có thể có xu hướng zig zag nào đó. Rất khó để thấy ra.

Thử tách xu hướng đó bằng B-spline. B-spline chia toàn khoảng của predictor variable thành khoảng nhỏ, gán một parameter cho mỗi khoảng. Những parameter này dần dần mở và tắt để tổng của chúng thành đường zigzag. Mục tiêu cuối cùng là tạo đường zigzag từ những hàm số ít zigzag hơn.

Đây là cách giải thích dài hơn, với ví dụ minh hoạ. Mục tiêu của ta là ước lượng xu hướng nở hoa từ hàm số zigzag. Đầu tiên, giống như polynomial, ta tạo biến mới và cho vào linear model, $\mu_i$. Khác với polynomial, B-spline không transform biến bằng bình phương hay lập phương. Mà nó tạo một dãy variable mới tự chế. Mỗi variable này dùng để mở hoặc tắt parameter trong khoảng cụ thể của predictor variable thực. Mỗi variable tự chế này là **HÀM CƠ BẢN BASIS**. Model này nhìn rất quen:

$$\mu_i = \alpha + w_1B_{i,1} + w_2B_{i,2} + w_3B_{i,3} + ... $$

$B_{i,n}$ là hàm cơ bản thứ *n* tại dòng *i*, *w* là trọng số của hàm cơ bản đó. Những *w* này giống như slope, tuỳ chỉnh ảnh hưởng của mỗi hàm cơ bản lên trung bình $\mu_i$. Cho nên đây cũng là linear regression, nhưng với những variable tự chế.

![](/assets/images/figure 4-21.png)

Nhưng ta làm thế nào để tạo basis B? Trường hợp đơn giản nhất trong hình trên là chia toàn khoảng của data thành 4 phần bằng nhau, thể hiện bằng 5 dấu thắt "+" ở trên hình, nhứng "+" nằm ở khoảng tứ phân vị của data. Bởi vì có ít data ở quá khứ xa xôi nên khoảng tứ đầu tiên khá rộng.

Năm điểm này là 5 hàm basis khác nhau, là B. Những variable tự chế này dùng để chuyển đổi nhẹ nhàng từ vùng trước sang vùng sau. Ở Basis đầu tiên sẽ có giá trị là 1, basis còn lại là 0. Khi di chuyển dần ra bên phải thì basis 1 giảm dần giá trị và basis 2 tăng dần. Tại dấu thắt 2, basis 2 có giá trị 1, basis khác là 0.

Một tính chất của hàm basis là nó ảnh hưởng khu trú parameter. Tại một điểm bất kỳ ở trục hoành, chỉ hai hàm basis có giá trị non-zero. Ví dụ ở năm 1200, basis 1 và 2 là non zero. Vậy parameter của basis 1 và 2 là parameter ảnh hưởng duy nhất vào năm 1200. Nó khác với polynomial là parameter ảnh hưởng toàn bộ hình dáng đường cong.

Hình ở giữa là basis nhân với weight tương ứng. Những weight này là từ kết quả của việc fit data vào model. Tôi sẽ hướng dẫn sau. Trọng số weight có thể dương hoặc âm. Ví dụ như basis 5 xuống thấp hơn dưới zero. Nó có weight âm. Để tạo dự báo cho bất kỳ năm nào, ví dụ như năm 1200, ta chỉ cần cộng tất cả những hàm số basis nhân weight. Tổng của chúng hơi cao hơn zero.

Ở hình cuối, tôi vẽ đường spline, là khoảng tin cậy 97% của $\mu$, dựa vào data thô. Bạn có thể đoán hình này phản ảnh thời tiết toàn cầu như thế nào. Nhưng có nhiều thứ còn trong data, trước năm 1800. Để nhìn thấy rõ hơn, ta làm 2 thứ. Hoặc dùng nhiều nút thắt hơn, càng nhiều thì spline càng linh hoạt. Hoặc thay vì dùng ước lượng linear, ta có thể dùng bậc mũ cao hơn.

Bây giờ ta sẽ dùng code, code này cho phép ta thay đổi lượng nút thắt và bậc mà ta thích. Bước tiếp theo là chọn bậc, nó quyết đinh cách mà các hàm basis gộp lại, hay sự tương tác giữa các parameter để tạo spline. Với bậc 1, hai hàm cộng lại tại mỗi điểm bất kỳ. Với bậc 2, có 3 hàm cộng lại tại mỗi điểm. Bậc 3 có 4 điểm. 

```python
d2 = d[~d['doy'].isnull()][["year","doy"]]  # lấy những năm có doy
num_knots = 15
degree = 3

knots = d2['year'].quantile(np.linspace(0,1,num_knots)).to_list()
knots = np.pad(knots, (degree, degree), mode="edge")

B = BSpline(knots, np.identity(num_knots+degree-1), k=degree)(d2['year'])

plt.figure(figsize=(12,5))
plt.subplot(xlim=(d2.year.min()-100, d2.year.max()+100), ylim=(0, 1),
            xlabel="year", ylabel="basis value")
for i in range(B.shape[1]):
    plt.plot(d2.year, B[:, i], "k", alpha=0.5)
plt.scatter(knots, np.ones(len(knots))-0.05, marker="+")
plt.title(f'Degree= {degree}')
```

![](/assets/images/figure 4-22.svg)

Ta có `B` là matrix các basis với 15 knots và degree 3, chứa 827 dòng và 17 cột, tương ứng với số năm. Một cột là một hàm basis.

Để lấy được weight của mỗi hàm số basis, ta dùng model linear regression. Ta xem mỗi cột của `B` như là một variable, ta có *intercept* để lấy ngày nở hoa trung bình. Model được định nghĩa như sau:

$$ \begin{matrix}
D_i & \sim \text{Normal}(\mu_i. \sigma) \\
\mu_i & = \alpha + \sum_{k=1}^{K} w_k B_{k,i}\\
\alpha & \sim \text{Normal} (100,10)\\
w_j & \sim \text{Normal} (0,10)\\
\sigma & \sim \text{Exponential}(1) 
\end{matrix}$$

Model này có vẻ lạ, nhưng những gì nó làm là nhân mỗi giá trị basis với parameter $w_k$ tương ứng.

Đây là làn đầu tiên ta dùng phân phối exponetial làm prior. Nó rất có ích khi làm prior cho param scale (sigma), vì ta cần nó luôn dương. Prior của sigma có rate là 1. Ta có thể đọc phân phối exponential như là nó chứa thông tin không gì khác ngoài độ lệch trung bình. Trung bình là đảo ngược của rate. Vậy trong trường hợp này là trung bình = 1/1 = 1. Nếu rate là 0.5, thì trung bình là 2. Ta sẽ dùng phân phối prior exponential rất nhiều trong sách, thay thế cho uniform. Thông thường ta nghĩ về độ lệch chuẩn trung bình hơn là maximum.

Để fit model ta phải tính tổng của weight nhân với toàn bộ basis. Cách đơn giản nhất là dùng phép nhân ma trận.

```python
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10), sample_shape=B.shape[1:])
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + B @ w
    numpyro.sample("mu", dist.Delta(mu), obs=mu)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)

start = {"w": np.zeros(B.shape[1])}
m4_7 = AutoLaplaceApproximation(model, init_strategy=init_to_value(start))
svi = SVI(model, m4_7, optim.Adam(1), AutoContinuousELBO(),
          B=B, D=d2['doy'].values)
init_state = svi.init(PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, np.zeros(10000))
p4_7 = svi.get_params(state)
```

Mặc dù ta có thể đọc bảng giá trị biên nhưng nó rất khó hiểu, thay vì vậy ta sẽ plot kết quả lên.

```python
post = m4_7.sample_posterior(PRNGKey(1), p4_7, (1000,))
w = np.mean(post["w"], 0)

plt.figure(figsize=(10,5))
plt.subplot(xlim=(d2.year.min()-100, d2.year.max()+100), ylim=(-8, 8),
            xlabel="year", ylabel="basis * weight")
for i in range(B.shape[1]):
    plt.plot(d2['year'], w[i] * B[:, i], "k", alpha=0.5)
plt.scatter(knots, np.ones(len(knots))*7.5, marker="+")
```

![](/assets/images/figure 4-23.svg)

Cộng thêm 97% của khoảng tin cậy của $\mu$ và các predictive values.

```python
plt.figure(figsize=(10,3))
mu = Predictive(m4_7.model, post, return_sites=["mu"])(
    PRNGKey(2), B, None)["mu"]
mu_PI = np.percentile(mu, q=(1.5, 98.5), axis=0)
az.plot_pair(d2[["year", "doy"]].astype(float).to_dict(orient="list"),
             scatter_kwargs={"c": "royalblue", "alpha": 0.8}, ax=plt.gca())
plt.fill_between(d2.year, mu_PI[0], mu_PI[1], color="k", alpha=0.5);
```

![](/assets/images/figure 4-24.svg)

Có gì đó xảy ra vào năm 1500. Nếu bạn thêm nhiều knots hơn thì đường này sẽ zigzag hơn. Bao nhiêu knots là đủ, câu hỏi này sẽ trả lời chương sau. Nhưng chú ý rằng đây không tính nhân quả. Để tìm hiểu sâu hơn, bạn phải so sánh xu hướng này với nhiệt độ thu thập để giải thích.

### 4.5.3 Nhưng hàm làm mượt khác của thế giới thực.

Spline ở phần trước chỉ mới bắt đầu. Một lớp model, gọi là **GENERALIZED ADDICTIVE MODELS (GAMs)**, tập trung vào dự báo một biến outcome dựa trên hàm làm mượt của vài biến. Chủ đề rất sâu đến nổi cần một sách riêng.

## <center>4.6 Tổng kết</center>

Chương này giới thiệu model linear regression đơn giản, là một khung quy trình để liên quan predictor variable với outcome variable. Phân phối Gaussian tạo thành likelihood của model này, bởi vì nó đếm số lượng tương đối số cách kết hợp khác nhau của trung bình và độ lệch chuẩn để tạo ra quan sát.

Để fit data vào model, chương này giới thiệu phương pháp quadratic approximation, và cũng như các phương pháp vẽ đồ hoạ.

Chương sau giới thiệu khái niệm multivariate linear regression. Kỹ thuật trong chương này là cơ sở để học tiếp các chương tiếp sau.