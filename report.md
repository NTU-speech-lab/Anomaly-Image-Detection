<center><font size="30"><b>ML HW10</b></font></center>

<center><span style="font-weight:light; color:#7a7a7a; font-family:Merriweather;">by b06902034 </span><span style="font-weight:light; color:#7a7a7a; font-family:Noto Serif CJK SC;">黃柏諭</span></center>

---

### Problem 1

*   baseline (fcn)

    <img src="/run/user/1000/gvfs/sftp:host=140.112.246.89,port=987/home/alec/Documents/ML/hw10-alechuang98/img/fcn_p1.png" style="zoom:67%;" />

*   best (cnn)

    <img src="/run/user/1000/gvfs/sftp:host=140.112.246.89,port=987/home/alec/Documents/ML/hw10-alechuang98/img/cnn_p1.png" style="zoom:67%;" />

### Problem 2

根據觀察Kaggle上的AUC score結果，使用n_component = 3。

|      | Baseline |  Best   |
| ---- | :------: | :-----: |
| KNN  | 0.59977  | 0.61619 |
| PCA  | 0.62752  | 0.55873 |

### Problem 3

*   original

    <img src="/run/user/1000/gvfs/sftp:host=140.112.246.89,port=987/home/alec/Documents/ML/hw10-alechuang98/img/origin_p3.png" style="zoom:67%;" />

*   baseline

    <img src="/run/user/1000/gvfs/sftp:host=140.112.246.89,port=987/home/alec/Documents/ML/hw10-alechuang98/img/baseline_p3.png" style="zoom:67%;" />

*   best

    <img src="/run/user/1000/gvfs/sftp:host=140.112.246.89,port=987/home/alec/Documents/ML/hw10-alechuang98/img/best_p3.png" style="zoom:67%;" />

    
    
 ### Problem 4

*   f1 score需要找出明確的threshold，ROC-AUC score則不需要，可以專注在model改進上。
*   ROC-AUC score能夠平等的重視兩個class的分類表現，f1 score較適合在skew data set作為評分標準，例如罕見疾病的辨識等。

