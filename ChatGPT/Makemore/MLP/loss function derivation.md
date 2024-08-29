



#### Cross Entropy Loss :


$$
p_i = \dfrac{e^{l_i}}{\sum_je^{l_j}}
$$
et
$$
\dfrac{d}{dl_i}\bigg[-\ln\dfrac{e^{l_y}}{\sum_je^{l_j}}\bigg]

= -\frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \dfrac{d}{dl_i}\dfrac{e^{l_y}}{\sum_je^{l_j}}
$$
if $i \neq y$ : 
$$
-\frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \dfrac{d}{dl_i}\dfrac{e^{l_y}}{\sum_je^{l_j}}

= -\frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \Bigg[\dfrac{0 \cdot \sum_je^{l_j} - e^{l_y}e^{l_i}}{\big(\sum_je^{l_j}\big)^2}\Bigg]

= \frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \dfrac{e^{l_y}e^{l_i}}{\big(\sum_je^{l_j}\big)^2}

= \dfrac{e^{l_i}}{\sum_je^{l_j}}

= p_i
$$
if $i = y$ :
$$
-\frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \dfrac{d}{dl_i}\dfrac{e^{l_y}}{\sum_je^{l_j}}

= -\frac{\sum_je^{l_j}}{{e^{l_y}}} \cdot \Bigg[\dfrac{e^{l_y}}{\sum_je^{l_j}} - \dfrac{e^{l_y}e^{l_i}}{\big(\sum_je^{l_j}\big)^2}\Bigg]

= -1 + \dfrac{e^{l_i}}{\sum_je^{l_j}} = p_i - 1
$$


#### BatchNorm Layer :

$$
\mu_B = \dfrac{1}{m}\sum_{i=1}^mx_i \\

\sigma_B^2 = \dfrac{1}{m-1}\sum_{i=1}^m(x_i - \mu_B)^2 \\

\hat{x}_i = \dfrac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\

y_i = \gamma\hat{x}_i + \beta
$$

We have
$$
\dfrac{dL}{dy_i}
$$
but we seek
$$
\dfrac{dL}{dx_i} = \: ???
$$
First :
$$
\dfrac{dL}{d\hat{x}_i} = \gamma\dfrac{dL}{dy_i}
$$
Second :
$$
\dfrac{dL}{d\sigma_B^2} = -\dfrac{\gamma}{2}\sum_{i=1}^m(x_i - \mu_B)(\sigma_B^2 + \epsilon)^{-3/2}\dfrac{dL}{dy_i}
$$
Summation because $\sigma_B^2$ is spread to all of the $\hat{x}_i$ matrix, so distributed to all its elements.

Third :
$$
\dfrac{dL(\sigma_B^2)}{d\mu_{B,1}} = -\dfrac{2\gamma}{m-1}\sum_{i=1}^m(x_i - \mu_B)\dfrac{dL}{dy_i} = -\dfrac{2\gamma}{m-1}(m\mu_B - m\mu_B)\dfrac{dL}{dy_i} = 0
$$

$$
\dfrac{dL(\hat{x}_i)}{d\mu_{B,2}} = -\gamma\sum_{i=1}^m\dfrac{1}{\sqrt{\sigma_B^2 + \epsilon}}\dfrac{dL}{dy_i} = -\gamma\sum_{i=1}^m(\sigma_B^2 + \epsilon)^{-1/2}\dfrac{dL}{dy_i}
$$

from which
$$
\dfrac{dL}{d\mu_B} = -\gamma\sum_{i=1}^m(\sigma_B^2 + \epsilon)^{-1/2}\dfrac{dL}{dy_i}
$$
Summation for $\mu_B$ for the same reason as the one for $\sigma_B^2$.

Fourth :
$$
\dfrac{dL}{dx_i} = \dfrac{dL}{d\hat{x}_i}\dfrac{d\hat{x}_i}{dx_i} + \dfrac{dL}{d\mu_B}\dfrac{d\mu_B}{dx_i} + \dfrac{dL}{d\sigma_B^2}\dfrac{d\sigma_B^2}{dx_i}
$$
with
$$
\dfrac{d\mu_B}{dx_{i}} = \dfrac{1}{m}
$$

$$
\dfrac{d\sigma_B^2}{dx_i} = \dfrac{2}{m-1}(x_i - \mu_B)
$$

$$
\dfrac{d\hat{x}_i}{dx_i} = (\sigma_B^2 + \epsilon)^{-1/2}
$$

from which
$$
\dfrac{dL}{dx_i} = \gamma(\sigma_B^2 + \epsilon)^{-1/2}\dfrac{dL}{dy_i} - \gamma\sum_{j=1}^m(\sigma_B^2 + \epsilon)^{-1/2}\dfrac{1}{m}\dfrac{dL}{dy_j} - \dfrac{\gamma}{2}\sum_{j=1}^m(x_j - \mu_B)(\sigma_B^2 + \epsilon)^{-3/2}\dfrac{2}{m-1}(x_i - \mu_B)\dfrac{dL}{dy_i} \\ \\

= \gamma(\sigma_B^2 + \epsilon)^{-1/2}\Bigg[\dfrac{dL}{dy_i} - \dfrac{1}{m}\sum_{j=1}^m\dfrac{dL}{dy_j} - \dfrac{1}{m-1}\sum_{j=1}^m\dfrac{(x_j - \mu_B)}{\sqrt{\sigma_B^2 + \epsilon}}\dfrac{(x_i - \mu_B)}{\sqrt{\sigma_B^2 + \epsilon}}\dfrac{dL}{dy_j}\Bigg] \\ \\

= \gamma(\sigma_B^2 + \epsilon)^{-1/2}\Bigg[\dfrac{dL}{dy_i} - \dfrac{1}{m}\sum_{j=1}^m\dfrac{dL}{dy_j} - \dfrac{\hat{x}_i}{m-1}\sum_{j=1}^m\hat{x}_j\dfrac{dL}{dy_j}\Bigg]

= \dfrac{\gamma(\sigma_B^2 + \epsilon)^{-1/2}}{m}\Bigg[m\dfrac{dL}{dy_i} - \sum_{j=1}^m\dfrac{dL}{dy_j} - \dfrac{m}{m-1}\hat{x}_i\sum_{j=1}^m\hat{x}_j\dfrac{dL}{dy_j}\Bigg]
$$


