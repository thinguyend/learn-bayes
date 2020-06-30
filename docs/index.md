# Giới thiệu Bayesian statistics

Trong vòng những năm gần đây, ta thấy ngày càng có nhiều ứng dụng dựa trên kỹ thuật *Machine Learning*, đặc biệt là trong lĩnh vực search engine, thương mại điện tử, quảng cáo, mạng xã hội, v,v,.. Những ứng dụng này tập trung vào độ chính xác trong dự đoán và cần lượng data lớn ( tính bằng tetrabyte). Thực tế đó là nền tảng của những ông Tech lớn như Google, Facebook,..

Tuy nhiên, đa số những ứng dụng này là *blackbox*, nghĩa là không thể diễn giải được. Ví dụ như model quảng cáo trúng đích, người quản lý không biết model hoạt động như thế nào, chỉ cần nó cho kết quả tốt là được.  
Một nhược điểm thứ 2 là những ứng dụng này cần rất nhiều data. Ví dụ như trong ứng dụng quảng cáo định hướng, họ cần hàng triệu người sử dụng để tạo ra model quảng cáo.  

Những giới hạn này làm cho việc tạo model khó hơn ở những lĩnh vực ít data hoặc chuyên sâu. Nó cũng có thể gây tác dụng phụ trong bối cảnh liên quan tới sinh mạng và luật pháp như y học hoặc bảo hiểm. Ở đây, model dự đoán với độ chính xác phải kèm theo độ tin cậy để ước lượng nguy cơ.  
Ví dụ: Chúng ta phải ước lượng được sự không chắc chắn khi đưa ra chẩn đoán có bệnh cho con người.

**Bayesian** là một phương pháp phân tích có thể vượt qua những nhược điểm này. Kỹ thuật bắt đầu với thiết lập niềm tin bản thân vào hệ thống cần model, sau đó kết hợp với data thu thập được để tạo ra model bị ràng buộc bởi niềm tin bản thân và data.  
Từ model đó, ta có thể dùng để dự đoán và kèm theo đó là một độ tin cậy được biểu diễn bằng phân phối.  
Phương pháp Bayesian hoạt động tốt khi có dữ liệu ít, kèm theo khái niệm độ tin cậy, và có thể diễn giải được.

**Probabilistic programming** là một dạng lập trình bậc cao, có ưu điểm che đậy các phép tính toán phức tạp trong phương pháp bayesian.  

## [Nội dung dịch từ sách Statistical Rethinking](./table-of-content)

# Bắt đầu với Probabilistic programming

## Cài đặt:
- [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html): Phiên bản nhẹ của anaconda, một ứng dụng để download python và các packages, tạo môi trường biệt lập để làm việc cho từng project.
- Sau khi cài đặt miniconda xong, bạn sẽ làm việc trong môi trường command promt trong windows hoặc terminal trong macos.
    - command promt: search cortana -> gõ `cmd` -> run
    - terminal: search spotlight -> gõ `terminal` -> run
    - gõ trong cmd `conda --version` để kiểm tra cài đặt thành công.
- Cài đặt channel conda-forge: [conda-forge](https://conda-forge.org) là một channel của conda chứa hầu như các packages có trong python, mà khi bạn cài đặt thì không cần quan tâm đến sự tương thích các packages với nhau.
```
conda config --add channels conda-forge 
conda config --set channel_priority strict 
```
- Tạo môi trường mới với \<env_name\> là tên bạn chọn với python=3.7: 
```
conda create --name <env_name> python=3.7
```
- Kích hoạt env:
```
conda activate <env_name>
```
- Cài đặt các packages sẽ dùng trong site này (sẽ update dần)
```
conda install pytorch numpyro arviz jupyterlab pandas causalgraphicalmodels daft
```
- Mở Jupyter và code python trong đó
```
jupyter lab
```

## Học Python cơ bản
Bạn cần phải biết Python. Nếu bạn mới bắt đầu thì có thể tìm hiểu cơ bản qua [tutorials trong python doc.](https://docs.python.org/3.7/tutorial/index.html)  

# Nguồn tài liệu tham khảo
- Statistical Rethinking - Richard McElreath 
- [rethinking-pyro](https://github.com/fehiepsi/rethinking-pyro) & [rethinking-numpyro](https://github.com/fehiepsi/rethinking-numpyro) - Du Phan

# Chúc các bạn thành công