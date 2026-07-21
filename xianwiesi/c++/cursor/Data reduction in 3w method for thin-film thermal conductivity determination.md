## Data reduction in $3 \omega$ method for thin-film thermal conductivity determination

T. Borca-Tasciuc; A. R. Kumar; G. Chen

Rev. Sci. Instrum. 72, 2139-2147 (2001)
https://doi.org/10.1063/1.1353189
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-01.jpg?height=415&width=1520&top_left_y=2237&top_left_x=448)

# Data reduction in $3 \omega$ method for thin-film thermal conductivity determination 

T. Borca-Tasciuc, ${ }^{\text {a) }}$ A. R. Kumar, and G. Chen ${ }^{\text {b) }}$<br>Mechanical and Aerospace Engineering Department, Nanoscale Heat Transfer and Thermoelectrics Laboratory, University of California at Los Angeles, Los Angeles, California 90095-1597

(Received 8 August 2000; accepted for publication 11 January 2001)


#### Abstract

The $3 \omega$ method has been proven to be very useful for determining the thermal conductivity of thin films and their substrates. Several simplifications are often used in determining the thermal conductivity of the films based on the experimentally measured $3 \omega$ signal. These simplifications, however, have limited range of applicability. In this work, we present a detailed analysis and mathematical modeling of the $3 \omega$ method applied for different experimental conditions. Effects considered include the finite substrate thickness, anisotropic nature of the film and substrate thermal conductivity, the film-substrate thermal property contrasts, the effect of heat capacitance of the heater, and the effect of thermal boundary resistance. Several experimental results are analyzed using the models presented. This work shows that the $3 \omega$ method can be extended to a wide range of sample conditions, with anisotropic conductivities in both the substrate and the film, and with small film-substrate conductivity contrast. © 2001 American Institute of Physics.


[DOI: 10.1063/1.1353189]

## I. INTRODUCTION

The $3 \omega$ method has been used extensively for measuring the thermal conductivity of thin films and bulk materials. ${ }^{1-15}$ In this method a thin electrically conductive wire is deposited onto the specimen whose thermal conductivity needs to be measured. The wire functions as both a heater and a temperature sensor. An ac current with angular modulation frequency $\omega$ is driven through the wire, causing Joule heating at a frequency of $2 \omega$. The generated thermal wave diffuses into the specimen and the penetration depth is determined by the thermal diffusivity of the specimen and the frequency of the ac current. Since the resistance of the heater is proportional to the temperature, the resistance will be modulated at $2 \omega$. The voltage drop along the wire thus contains a third harmonic that depends on the ac temperature rise of the heater and could be used to extract the thermal conductivity of the specimen. Thermal conductivities of thin films down to several nanometers thickness were measured using this technique. ${ }^{2-4,8-10,12}$ Moreover, the $3 \omega$ method was employed to measure the thermal conductivity of anisotropic thin films. ${ }^{5-7,13}$ For anisotropic thermal conductivity measurements, the combination between the heater wire width and the film thickness determine the measurement sensitivity to the in-plane and cross-plane thermal properties of the film. Choosing a heater width much larger than the film thickness, the measured temperature drop could be assumed to be sensitive mainly to the cross-plane thermal conductivity of the film. If the wire width is smaller or comparable to the film thickness, the heat produced in the heater wire tends to

[^0]spread inside the film and the measured temperature signal is influenced by both the in-plane and cross-plane thermal conductivities of the film.

Of crucial importance to the accuracy of the thermal conductivity determination are the heat conduction models used to obtain the substrate and the thin-film thermal conductivities. An approximate analytical expression is often employed to determine the thermal conductivity of the substrate from the slope of the real part of the ac temperature expressed as a function of the logarithm of modulation frequency. ${ }^{1-3,9}$ The corresponding temperature drop developed in the substrate may be then calculated. For measuring the thermal conductivity of a thin-film deposited onto the substrate, the temperature drop across the thin film is inferred from the difference between the measured total temperature rise and the calculated temperature drop across the substrate. Using a simple one-dimensional heat conduction analysis, the thermal conductivity of the thin film can be determined.

These simplified analytical models are subjected to a variety of constraints that are not well documented in literature. In practice, some of these constraints may not be always satisfied. For example, one condition for using the onedimensional steady-state heat conduction model to determine the thermal conductivity of thin films deposited on the substrate is that the film thermal conductivity must be much smaller than the substrate thermal conductivity. This condition imposes a severe limitation to the range of samples that the method can handle.

In this article, we present a detailed analysis of the $3 \omega$ data reduction methods. We start from a two-dimensional heat conduction model applied to the general case of a multilayer film-on-substrate system with anisotropic thermophysical properties. The principal heat conduction axes are considered in the directions parallel and perpendicular to the film plane. We then focus on several particular situations
including anisotropic substrate, one film on a substrate, and two films on a substrate. It is shown that using the differential method to measure the temperature drop across the film of interest can reduce the uncertainty in the determination of the thermal conductivity of the film. Approximate formulations for the heater temperature rise are derived for the case of anisotropic substrate. Moreover, a steady-state formulation applied to the case of one-film-on-substrate system is used to discuss the effects due to two-dimensional heat conduction in the film and film/substrate thermal conductivity contrast. Experimental data are presented to validate the model analysis.

## II. GENERAL SOLUTION FOR THE 2D HEAT CONDUCTION ACROSS A MULTILAYER-FILM-ONSUBSTRATE SYSTEM

The $3 \omega$ method is often employed in the thermal conductivity characterization of film-on-substrate systems using a simple one-dimensional heat conduction analysis to determine the thermal conductivity of the thin film and the slope method to determine the thermal conductivity of the substrate. ${ }^{1-3}$ However, thermal characterization sometimes is needed for samples that have more than one film deposited onto the substrate, ${ }^{4-6,8,11-13}$ for films and substrates that have anisotropic thermophysical properties, ${ }^{5-7,13-15}$ and for substrates that may not be considered semi-infinite. ${ }^{14}$ A solution for the heater temperature rise on a film-on-semi-infinitesubstrate structure with anisotropic thermal properties of the film was derived in Ref. 5. Kim et al. ${ }^{9}$ investigated the case of an isotropic multilayer film-on-finite-substrate structure. In this section, the expression for the heater temperature rise is derived in the general case of a multilayer film-on-finite/ semi-infinite-substrate system with anisotropic thermophysical properties. The derivation is based on a two-dimensional heat-conduction model across the system and a uniform heat flux boundary condition between the heater and the top film. The integral Fourier transformation technique ${ }^{16}$ is employed to obtain the analytical solution. Neglecting the contributions from the thermal mass of the heater and thermal boundary resistances, effects discussed later in the article, the complex temperature rise of a heater dissipating $p / l(\mathrm{~W} / \mathrm{m})$ peak electrical power per unit length is

$$
\begin{equation*}
\Delta T=\frac{-p}{\pi l k_{y_{1}}} \int_{0}^{\infty} \frac{1}{A_{1} B_{1}} \frac{\sin ^{2}(b \lambda)}{b^{2} \lambda^{2}} d \lambda \tag{1}
\end{equation*}
$$

where

$$
\begin{align*}
& A_{i-1}=\frac{A_{i} \frac{k_{y i} B_{i}}{k_{y_{i-1}} B_{i-1}}-\tanh \left(\varphi_{i-1}\right)}{1-A_{i} \frac{k_{y i} B_{i}}{k_{y_{i-1}} B_{i-1}} \tanh \left(\varphi_{i-1}\right)}, \quad i=2 \ldots n  \tag{2}\\
& B_{i}=\left(k_{x y i} \lambda^{2}+\frac{i 2 \omega}{\alpha_{y i}}\right)^{1 / 2}  \tag{3}\\
& \varphi_{i}=B_{i} d_{i}, \quad k_{x y}=k_{x} / k_{y} \tag{4}
\end{align*}
$$

In the above expressions, $n$ is the total number of layers including the substrate, subscript $i$ corresponds to the $i$ th
layer starting from the top, subscript $y$ corresponds to the direction perpendicular to the film/substrate interface (cross plane), $b$ is the heater half width, $k$ is the thermal conductivity of the layer, $\omega$ is the angular modulation frequency of the electrical current, $d$ is the layer thickness, and $\alpha$ is the thermal diffusivity. The effect of the thermal conductivity anisotropy is introduced through the term $k_{x y}$, which is the ratio of the in-plane to cross-plane thermal conductivity of the layer. For the substrate layer $i=n$, and if the substrate is semiinfinite $A_{n}=-1$. When the substrate has a finite thickness, the value of $A_{n}$ depends on the boundary condition at the bottom surface of the substrate: $A_{n}=-\tanh \left(B_{n} d_{n}\right)$ for an adiabatic boundary condition or $A_{n}=-1 / \tanh \left(B_{n} d_{n}\right)$ if the isothermal boundary condition is considered.

In subsequent sections of this article, Eq. (1) is employed to calculate the heater temperature rise for different specimen configurations, to estimate the range of applicability of the simplified models currently used in the data reduction, and to correct the relative errors introduced by these models in the thermal conductivity determination. Furthermore, the discussion is illustrated with experimental examples appropriate for each studied situation.

## III. SUBSTRATE THERMAL CONDUCTIVITY MEASUREMENT FOR ANISOTROPIC AND FINITE THICKNESS CASE

Cahill and co-authors ${ }^{2,3}$ derived a formula for the ac temperature rise of a heater deposited on an isotropic substrate, taking into account the finite width of the heat source and the temperature averaging across the width of the metal line. The solution obtained by Cahill ${ }^{2}$ is identical to the temperature rise given in Eq. (1) in the particular case of a semiinfinite isotropic substrate with no film. If the penetration depth $q^{-1}=\sqrt{\alpha_{S} / 2 \omega}$ (where subscript $S$ is associated with substrate) is further considered much larger than the heater half width:

$$
\begin{equation*}
q^{-1} \gg b \tag{5}
\end{equation*}
$$

the heater can be approximated as a line source, and the complex temperature rise of the heater can be approximated as ${ }^{2,3}$

$$
\begin{align*}
T_{S} & =\frac{p}{\pi l k_{S}}\left(0.5 \ln \left\{\frac{\alpha_{S}}{b^{2}}\right\}-0.5 \ln \{\omega\}+\eta\right)-i\left(\frac{p}{4 l k_{S}}\right) \\
& =\frac{p}{\pi l k_{S}} f_{\text {linear }}(\ln \omega) \tag{6}
\end{align*}
$$

where $\eta$ is a constant and $f_{\text {linear }}$ is a linear function of $\ln \omega$. Equation (6) shows how the thermal conductivity of an isotropic substrate can be determined from the slope of the real part of the ac temperature amplitude as a function of the logarithm frequency. Subsequently, we will refer to the substrate thermal conductivity measurement technique based on Eq. (6) as the "slope method" and we will refer to the slope of the real part of the temperature amplitude as a function of logarithm frequency as simply the "slope." By normalizing the temperature rise with $\left(-p / \pi l k_{S}\right)$, Eq. (6) leads to a constant slope value of 0.5 for the normalized temperature rise. Another condition that limits the use of Eq. (6) for substrate
thermal conductivity measurement is the assumption of semi-infinite medium, which holds if the thermal penetration depth is smaller than $d_{S}$, substrate thickness; ${ }^{2}$

$$
\begin{equation*}
q^{-1} \ll d_{S} . \tag{7}
\end{equation*}
$$

The line source and semi-infinite substrate conditions expressed in Eqs. (5) and (7) are rather qualitative. One purpose of this article is to provide more quantitative criteria for the applicability of the slope method. Furthermore, Eq. (6) considers only the case of isotropic substrates although in some experimental situations the substrate thermal conductivity can be anisotropic. To address these issues we obtained an analytical expression of the ''slope,'' using Eq. (1) for the case of a finite anisotropic substrate with adiabatic boundary condition on the bottom surface. Dividing the analytical slope to ( $-p / \pi l k_{S} \sqrt{k_{S x y}}$ ) we obtained the following general expression for the normalized slope

$$
\begin{align*}
& \frac{d \theta / d \ln (\omega)}{-p /\left(\pi l k_{S y} \sqrt{k_{S x y}}\right)} \\
& \quad=0.5 \int_{0}^{\infty} \frac{z^{2}}{B_{S}^{3} \tanh \left(B_{S} \beta_{S}\right)} \\
& \quad \times\left(1+\frac{4 B_{S} \beta_{S}}{\left(e^{B_{S} \beta_{S}}+e^{-B_{S} \beta_{S}}\right)^{2} \tanh \left(B_{S} \beta_{S}\right)}\right) \frac{\sin ^{2}(\lambda)}{\lambda^{2}} d \lambda \tag{8}
\end{align*}
$$

where, $z=q b / \sqrt{k_{S x y}}, \beta_{S}=\sqrt{k_{S x y}}\left(d_{S} / b\right)$, and $B_{S}=\sqrt{\lambda^{2}+z^{2}}$. The nondimensional parameters $\beta_{S}$ and $z$ are related to three effects which have an influence on the normalized slope in the general case: the finite substrate effect expressed by $z$ and the $d_{S} / b$ ratio, the line-source approximation related to the $q b$ product, and the effect of substrate anisotropy described by $k_{x y}$. Figure 1(a) shows the normalized slope from Eq. (8) calculated as a function of the $z$ parameter for different values of $\beta_{S}$. The range of applicability of the slope method is estimated comparing the values predicted by Eq. (8) with the approximate normalized slope value of 0.5 determined from Eq. (6). For the case of a semi-infinite substrate ( $\beta_{S} \rightarrow \infty$ ), the approximate normalized slope is within $1 \%$ of the normalized slope from Eq. (8) if

$$
\begin{equation*}
z<0.2 \tag{9}
\end{equation*}
$$

Equation (9) indicates that the line-source approximation is valid (within $1 \%$ error in the normalized slope value) if the thermal penetration depth is at least 5 times larger than the ratio between the heater half width and the square root of substrate anisotropy. In this case, a simplified expression similar to the one in Eq. (6) can be derived for an anisotropic substrate;

$$
\begin{align*}
T= & \frac{p}{\pi l k_{S y} \sqrt{k_{S x y}}}\left[0.5 \ln \left(\frac{\alpha_{S y} k_{S x y}}{b^{2}}\right)-0.5 \ln \{\omega\}+\eta\right] \\
& -i\left(\frac{p}{4 l k_{S y} \sqrt{k_{S x y}}}\right) \tag{10}
\end{align*}
$$

If $z>0.2$, the line-source approximation is less applicable and the deviation between the approximate and exact solu-
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-04.jpg?height=1160&width=714&top_left_y=163&top_left_x=1161)

FIG. 1. Normalized slope [Eq. (8)], compared to the approximate normalized slope of 0.5 , is shown for different $\beta_{S}$ values: (a) as a function of the nondimensional parameter $z$, and (b) as a function of the nondimensional parameter $q d_{S}$. The line source and semi-infinite substrate approximations can be used with less than $1 \%$ error if $z<0.2$ and $q d_{S}>5$.
tions for the normalized slope increases rapidly as $z$ increases. At $z=1$, for example, the relative error introduced by the approximate expression is already $\sim 23 \%$. If substrate thickness is finite, for small $z$ values the thermal wave is reflected by the surface of the substrate and the normalized slope can be very different from that predicted by the slope method. This behavior is indicated in Fig. 1(a) by several curves calculated using Eq. (8) for $\beta_{S}$ values between 0.1 and 10000 . The lower the $\beta_{S}$, i.e., the thinner is the substrate relative to the heater width, the higher is the minimum $z$ value for which the approximate normalized slope expression could be used without large errors due to the finite substrate effect. However, for very small $\beta_{S}$ the required minimum $z$ values becomes too large to satisfy the line-source approximation. A general condition to address the finite substrate effect could be found if the finite-substrate curves plotted in Fig. 1(a) are replotted as a function of a new nondimensional parameter relating the substrate thickness and the thermal diffusion length of the thermal wave, $q d_{S}=z \beta_{S}$. As shown in Fig. 1(b), the normalized slope curves calculated for various $\beta_{S}$ parameters overlap for small $q d_{S}$ and large $\beta_{S}$ values where the line-source approximation is still valid. Furthermore, the influence of the finite substrate thickness effect is minimized and a good agreement (within 1\%) between the approximate normalized slope value and the exact expression is obtained if
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-05.jpg?height=549&width=689&top_left_y=181&top_left_x=260)

FIG. 2. Experimental temperature rise (dots) and the fitting based on a two-dimensional heat conduction model (lines) for a nanochanneled alumina sample (see inset for sample configuration) with anisotropic thermal conductivity. The experiment is carried out for two heater widths and the fitted values for the cross- and in-plane thermal conductivity are respectively $k_{S y}=1.5 \mathrm{~W} / \mathrm{mK}$ and $k_{S x}=1.05 \mathrm{~W} / \mathrm{mK}$.

$$
\begin{equation*}
q d_{S}>5 \tag{11}
\end{equation*}
$$

Eq. (11) indicates that the semi-infinite substrate thickness approximation is valid (within $1 \%$ error in the normalized slope value) if the thermal penetration depth is at least 5 times smaller than the substrate thickness. The conditions expressed in Eqs. (9) and (11) provide more quantitative expressions for the range of applicability of the slope method in the general case of finite and anisotropic substrate and could be used to determine the best frequency range and heater width to be used with a given sample. Moreover, they indicate that if $\beta_{S}<25$ it is not possible to simultaneously minimize the substrate effects and still satisfy the line source approximation. Hence, if $\beta_{S}<25$ no frequency range exists where the slope method could be used with less than $1 \%$ deviation of the approximate normalized slope from the exact expression in Eq. (8).

An example of the $3 \omega$ method-based thermal conductivity characterization of anisotropic and finite thickness substrate samples is the thermal conductivity measurement of nanochanneled alumina. ${ }^{14}$ The samples have regular nanochannels with axis aligned in the direction perpendicular to the template surface, as shown in the inset of Fig. 2. Electrical heaters/temperature sensors between 5 and $40 \mu \mathrm{~m}$ widths are deposited on the $60 \mu \mathrm{~m}$ thick alumina substrate. ${ }^{14}$ The thermal conductivity anisotropy of the sample was determined to be $\sim 0.7$. Since $\beta_{S}<25$ for both the 5 and $40 \mu \mathrm{~m}$ width heaters ( $\beta_{S} \sim 10$ for the small width heater and $\beta_{S}$ $\sim 1$ for the large width heater), the slope method based on the simplified expression from Eq. (10) could not be used to determine the thermal conductivity of the substrate. The thermal conductivity of the anisotropic alumina substrate was determined by fitting the experimental temperature rise with predictions based on the exact solution from Eq. (1). Temperature rise calculations based on the exact solution show that large width heaters are more sensitive to the cross-plane (along axis) thermal conductivity of the sample because the heat transport is mainly in the axial direction. Furthermore, small width heaters experience relatively more spreading of the heat in the in-plane direction and are sensitive to both the
in-plane (perpendicular to nanochannel axis) and cross-plane components of the thermal conductivity. Hence, the 5 and 40 $\mu \mathrm{m}$ width heaters and an iterative technique were used to determine the in-plane and cross-plane thermal conductivity of the substrate. Figure 2 shows the experimental temperature rise collected at room temperature along with the fit from Eq. (1) for both heaters. The determined anisotropic thermal conductivity of the nanochanneled alumina sample at room temperature is, $k_{S y}=1.5$ and $k_{S x}=1.05 \mathrm{~W} / \mathrm{mK}$.

## IV. THIN FILM ON SUBSTRATE THERMAL CONDUCTIVITY MEASUREMENT

Estimating the temperature drop across the film is required to determine the thin-film thermal conductivity. In the $3 \omega$ method, ${ }^{2,3}$ the temperature drop across the film is inferred from the difference between the experimental temperature rise of the heater and the calculated temperature rise of the bare substrate sample. The substrate thermal conductivity is determined from the slope method applied to the experimental temperature rise of the heater and usually a onedimensional heat conduction model is assumed across the film. With these approximations, the experimental temperature rise of the heater on the film plus substrate system can be written as ${ }^{2,3}$

$$
\begin{equation*}
T_{S+F}=T_{S}+\frac{p d_{F}}{2 b l k_{F}} \tag{12}
\end{equation*}
$$

where subscript $F$ denotes film properties and $T_{S}$ is calculated based on Eq. (6).

A different way to estimate the temperature drop across the film is to measure it experimentally by a differential technique. The differential $3 \omega$ method involves the experimental measurement of the temperature-rise difference between similar heaters deposited onto the specimen and a reference sample without the film of interest. ${ }^{6,8,10-13}$

The next sections discuss the relative errors in film thermal conductivity measurement between the sloped-based $3 \omega$ method and the differential $3 \omega$ method, the errors introduced by two-dimensional heat conduction effects and the filmsubstrate thermal conductivity contrast, and the influence of the heat capacitance of the heater and the thermal resistance effect on the slope method.

## A. Comparison between differential and slope methods

This section presents a detailed experimental error analysis of the film thermal conductivity measurement for the differential $3 \omega$ technique and the sloped-based $3 \omega$ technique. For clarity, the one-dimensional heat conduction model across the films is assumed to be valid.

Considering the expression from Eq. (12), the film thermal conductivity as determined from the sloped-based $3 \omega$ method may be written as

$$
\begin{align*}
k_{F} & =\frac{p d_{F}}{2 b l}\left[\left(\frac{2 v_{3 \omega}}{C_{\mathrm{rt}} v_{1 \omega}}\right)-T_{S}\right]_{\text {ave }}^{-1} \\
& =\frac{p d_{F}}{2 b l}\left[\left(\frac{2 v_{3 \omega}}{C_{\mathrm{rt}} v_{1 \omega}}\right)+2 \text { slope } f_{\text {linear }}(\ln \omega)\right]_{\text {ave }}^{-1} \tag{13}
\end{align*}
$$

where $v_{3 \omega}$ and $v_{1 \omega}$ correspond to the third and first harmonic of the experimental voltage and $C_{\mathrm{rt}}$ is the temperature coefficient of resistance. In the second part of Eq. (13), the calculated complex temperature rise of the substrate is substituted from Eq. (6) and the substrate thermal conductivity $k_{S}$ is explicitly written as a function of the "slope" of the experimental signal. In order to determine the thermal conductivity of the film, the calculated temperature drop across the film is averaged over the experimental frequency range. If the system contains additional films, the thickness and thermal properties of these films must be known in order to subtract their contribution from the total experimental temperature rise.

In the differential $3 \omega$ technique, the thermal conductivity of the film is calculated using the average temperature rise difference experimentally measured at the same power input by similar heaters deposited on the specimen and a reference sample without the film of interest:

$$
\begin{equation*}
k_{F}=\frac{d_{F}}{2}\left[\left(\frac{2 v_{3 \omega} b l}{p C_{\mathrm{rt}} v_{1 \omega}}\right)_{R+F}-\left(\frac{2 v_{3 \omega} b l}{p C_{\mathrm{rt}} v_{1 \omega}}\right)_{R}\right]_{\text {average }}^{-1} \tag{14}
\end{equation*}
$$

where subscript $R+F$ corresponds to the structure that includes the sample film and the subscript $R$ refers to the reference structure. Although the differential technique requires two sets of measurements, the film thermal conductivity determined from Eq. (14) is insensitive to the substrate thermal conductivity, and the contributions from any additional films is nearly eliminated by the measurement on the reference sample.

In order to compare the uncertainty in film thermal conductivity estimated from Eqs. (13) and (14), error analysis is carried out for a film-on-substrate system with $k_{F}$ $=9.5 \mathrm{~W} / \mathrm{mK}, d_{F}=1 \mu \mathrm{~m}$, and $k_{S}=150 \mathrm{~W} / \mathrm{mK}$. Our typical experimental uncertainty for each parameter is: $2 \%$ for $V_{3 \omega}$, $0.07 \%$ for $V_{1 \omega}, 2 \%$ for heater resistance, $1 \%$ for $d_{F}, 2 \%$ for $C_{\mathrm{rt}}, 2 \%$ for slope and $3 \%$ for $b$. The differential technique requires input from two sets of experiments as compared to the slope method that requires only one set of experimental data. The additional uncertainty due to the higher number of experimental variables results in a higher uncertainty of the differential technique. For the studied film-on substrate system, the uncertainty in the estimated $k_{F}$ is $\sim 7 \%$ for the slope method and $\sim 20 \%$ for the differential technique. However, as shown in Eq. (13), the value of the slope can strongly affect the film thermal conductivity. Experimentally, we have found that the substrate thermal conductivity determined from the slope method may depend on the heater width and the film thickness (for silicon substrate the variation in substrate thermal conductivity determined using different heater widths is as large as $20 \%$ ). Similarly, Kim and co-authors ${ }^{9}$ reported that the substrate thermal conductivity from the slope method deviates up to $10 \%$ from the actual
value, and is also sensitive to the film thickness. These errors in substrate thermal conductivity measurement are related to the thermal resistance and thermal capacitance of the heater and film, and will be discussed in Sec. V. Furthermore, if the sample structure contains more than one film, the uncertainty due to the additional variables (such as the thickness and thermal conductivity for each film) may further increase the error associated with the sloped-based estimation of the film thermal conductivity. Assuming a substrate thermal conductivity error of $20 \%$ and starting with the first part of Eq. (13), the film thermal conductivity value determined from the slope-based $3 \omega$ method deviates by $\sim 40 \%$ from the true thermal conductivity of the film. This error is larger than the uncertainty of the differential $3 \omega$ method.

The discussion presented in this section was carried out under the assumption of one-dimensional heat conduction across the film. If the thermal conductivity of the film is strongly anisotropic and/or the contrast between the thermal conductivity of the film and substrate is small, then twodimensional effects in the film may result in large errors in the estimated values of $k_{F}$ from Eq. (13) or (14) and a twodimensional heat conduction model must be used to backup the film thermal conductivity. In this case, the temperature drop across the film could depend on the thermal properties of the substrate and of all the films in the system, therefore both the $3 \omega$ and the differential $3 \omega$ technique must take into consideration their contributions. The uncertainty analysis for the two-dimensional case becomes more complicated. However, the discussion carried out in this section suggests that a "calculated" temperature drop [Eq. (13)] would be more affected by films and substrate uncertainties than a "measured" temperature drop [Eq. (14)]. Our twodimensional simulations show that the uncertainties related to the substrate thermal conductivity and the presence of additional films would have relatively less effect on the differential $3 \omega$ method than on the slope-based $3 \omega$ method. The two-dimensional effects are discussed in detail in the following section.

## B. Effects of the film/substrate thermal conductivity contrast and heat spreading inside the film

The one-dimensional thin-film thermal conductivity reduction method as given by Eq. (13) or (14) requires that substrate thermal conductivity be much higher than the thermal conductivity of the film. This is easily understood because in the limiting case of a semi-infinite substrate and $k_{S}$ equal to $k_{F}$, the temperature drop across the film ( $\Delta T_{F}$ ) is zero. Furthermore, it is required to minimize the heat spreading effects inside the film, which are due to film anisotropy and the aspect ratio between heater width and film thickness. The purpose of this section is to discuss the validity of the one-dimensional heat conduction modeling across the film and to provide formula for compensating the heat spreading inside the film and the small thermal conductivity contrast between the film and the substrate. The starting point is Eq. (1) for the case of an anisotropic thin film on a semi-infinite isotropic substrate. The temperature drop across the film, in the limit that frequency $\omega$ approaches to zero, is
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-07.jpg?height=1193&width=703&top_left_y=168&top_left_x=250)

FIG. 3. (a) Ratio of the true thermal conductivity of the film to the onedimensional thermal conductivity, and (b) the ratio from (a) normalized to $C /\left(1+0.38 \beta_{F}\right)$, as a function of thermal conductivity contrast between the film and substrate.

$$
\begin{equation*}
\Delta T_{F}=T_{F+S}-T_{S}=\frac{p d_{F}}{2 b l k_{F y}} C S \tag{15}
\end{equation*}
$$

where $T_{F+S}$ is the temperature rise of the heater placed on a structure consisting of both the film and substrate and $T_{S}$ is the temperature rise due to the substrate only, $C$ is a factor that relates to the contrast between the film thermal conductivity, $k_{F x}$ and $k_{F y}$, and the substrate thermal conductivity, $k_{S}$ :

$$
\begin{equation*}
C=1-\frac{k_{F x} k_{F y}}{k_{S}^{2}} \tag{16}
\end{equation*}
$$

and $S$ addresses the effect of lateral heat spreading in the film

$$
\begin{equation*}
S=\frac{2}{\pi} \int_{0}^{\infty} \frac{\sin ^{2} \lambda}{\lambda^{3}} \frac{\tanh \left(\lambda \beta_{F}\right)}{\left[1+\left(\sqrt{k_{F y} k_{F x}} / k_{S}\right) \tanh \left(\lambda \beta_{F}\right)\right] \beta_{F}} d \lambda \tag{17}
\end{equation*}
$$

where $\beta_{F}=\sqrt{k_{F x y}}\left(d_{F} / b\right)$. Using Eq. (15), the ratio of the true thermal conductivity of the film to the approximated 1D thermal conductivity of the film can be expressed as

$$
\begin{equation*}
\frac{k_{F y}}{k_{1 \mathrm{D}}}=C S \tag{18}
\end{equation*}
$$

This quantity is plotted in Fig. 3(a) for different values of $\beta_{F}$ and as a function of film to substrate thermal conductivity
contrast ratio $\sqrt{k_{F y} k_{F x}} / k_{S}$. The figure shows that onedimensional heat conduction modeling is strictly valid only if $\beta_{F} \rightarrow 0$ (when $C \rightarrow 1$ ) and $\sqrt{k_{F y} k_{F x}} / k_{S} \rightarrow 0$ (when $S \rightarrow 1$ ). As a general trend, the larger is the thermal conductivity ratio between film and substrate, i.e., the smaller is the contrast between the film and the substrate, the larger is the error in the film thermal conductivity as determined from the 1D heat conduction modeling. Furthermore, this error increases as larger $\beta_{F}$ values amplify the heat spreading effects due to the film anisotropy and/or the high aspect ratio between film thickness and heater width. For example, at $\sqrt{k_{F y} k_{F x}} / k_{S}$ $=0.2$ the 1D approximation yields an error of $\sim 5 \%$ if $\beta_{F}$ $=0$, and the error is $\sim 15 \%$ if $\beta_{F}=0.1$. For large values of the substrate thermal conductivity compared to film thermal conductivity, $C \sim 1$ and Eq. (18) can be used to isolate the effect of pure heat spreading inside the film:

$$
\begin{equation*}
\frac{k_{F y}}{k_{1 \mathrm{D}}}=\frac{2}{\pi} \int_{0}^{\infty} \frac{\sin ^{2} \lambda}{\lambda^{3}} \frac{\tanh \left(\lambda \beta_{F}\right)}{\beta_{F}} d \lambda \tag{19}
\end{equation*}
$$

For $\beta_{F}<10$, the integral expression in Eq. (19) is approximated within $3 \%$ error by the simpler formula ( 1 $\left.+0.38 \beta_{F}\right)^{-1}$. Therefore, if the film thermal conductivity is much smaller than the substrate thermal conductivity, the heat spreading effect could be accounted for by the onedimensional heat conduction model if the width of the heater is replaced with a "corrected" width of $2 b+0.76 d_{F}$ $\times \sqrt{k_{F x y}}$. An almost similar expression, $\left(2 b+0.88 d_{F}\right)$, was used by Cahill et al., in Ref. 17 to determine the thermal conductivity of alumina films deposited on Co-cemented WC substrates. To estimate how effective are the simple spreading correction and the thermal conductivity contrast correction [from Eq. (16)] in compensating for the twodimensional heat conduction effects into the film, the curves from Fig. 3(a) are normalized to $C /\left(1+0.38 \beta_{F}\right)$ and replotted in Fig. 3(b). An ideal correction would yield a constant value of 1 , independent of $\beta_{F}$ or film to substrate thermal conductivity ratio. However, as shown in the Fig. 3(b), the simpler corrections cannot fully compensate for the twodimensional effects for larger $\beta_{F}$ and larger film to substrate thermal conductivity ratios.

An example of a thin-film thermal conductivity measurement carried out for a film-on-substrate system subjected to a significant film/substrate thermal conductivity contrast effect is given in reference. ${ }^{11}$ The cross-plane thermal conductivity measurement of a $1-\mu$ m-thick $50 \AA \mathrm{Bi} / 50 \AA \mathrm{Sb}$ superlattice deposited on (111) CdTe substrate was carried out in the temperature range between 80 and 300 K using the differential $3 \omega$ method and a two-dimensional heat conduction modeling across the film. The width of the heaters/temperature sensors deposited on the sample and the reference is $10 \mu \mathrm{~m}$. The corresponding $\beta_{F}$ value for the studied superlattice film is $\sim 0.1$, and in the absence of any film/substrate thermal conductivity contrast effect the heat spreading effect itself would be responsible for just a $5 \%$ error in the film thermal conductivity estimated from the 1D heat conduction model (see Fig. 3). The experimental results shown in Fig. 4 indicate that the substrate thermal conductivity is comparable to the superlattice thermal conductivity at the upper end of the
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-08.jpg?height=554&width=689&top_left_y=178&top_left_x=260)

FIG. 4. Experimental thermal conductivity of a $\mathrm{Bi} / \mathrm{Sb}$ superlattice and the CdTe substrate. The substrate thermal conductivity is comparable to the superlattice thermal conductivity at the upper end of the temperature range and a significant film/substrate thermal conductivity contrast effect is expected for this temperature range.
temperature range. According to calculations shown in Fig. 3, the significant film/substrate thermal conductivity contrast effect that is present close to room temperature would introduce large errors in the 1D film thermal conductivity determination. In order to illustrate this situation, Fig. 5 shows the experimentally measured heater temperature rise collected at 80 and 280 K for the superlattice and reference samples. The difference between the temperature rise of the superlattice and reference heaters at 280 K is $\leqslant 0$, yielding an unreasonable value for the one-dimensional thermal conductivity of the film. Therefore, two-dimensional heat conduction modeling similar to Eq. (15) must be used to compensate for the film/substrate thermal conductivity contrast effect. The lines in Fig. 5 are the predicted temperature rise of the heater calculated based on Eq. (1) for the fitted value of the crossplane thermal conductivity of the film at each temperature.

The heat spreading effect inside the film and the thermal conductivity measurement of anisotropic films is illustrated by the thermal characterization of a Ge quantum-dotsuperlattice structure grown on Si substrate and reported in
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-08.jpg?height=543&width=825&top_left_y=1845&top_left_x=124)

FIG. 5. Experimentally measured heater temperature rise (dots) collected at 80 and 280 K for the superlattice and reference samples. The difference between the temperature rise of the superlattice and reference heaters at 280 K is $\leqslant 0$, yielding an unreasonable value for the one-dimensional thermal conductivity of the film. The lines show the predicted temperature rise of the heater calculated based on Eq. (1) for the fitted value of the cross-plane thermal conductivity of the film at each temperature. 80 and 280 K for the superlattice and reference samples. The difference between the temperature rise of the superlattice and reference heaters at 280 K is $\leqslant 0$, yielding an unreasonable value for the one-dimensional thermal
conductivity of the film. The lines show the predicted temperature rise of the
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-08.jpg?height=562&width=690&top_left_y=180&top_left_x=1173)

FIG. 6. Experimental temperature rise (dots) measured by 30 and $2 \mu \mathrm{~m}$ width heaters deposited on the Ge-quantum-dot-superlattice and the reference samples. The experimental signal is compared to predictions for the temperature rise of the heaters (lines) calculated based on Eq. (1) for the fitted values of the in-plane and cross-plane thermal conductivity of the film.

Ref. 13. The thickness of the quantum-dots-superlattice is $\sim 1 \mu \mathrm{~m}$, and the measured in-plane and cross-plane thermal conductivities at room temperature are, respectively, 35 and $9 \mathrm{~W} / \mathrm{mK}$. The ratio $\sqrt{k_{F y} k_{F x}} / k_{S}$ is $\sim 0.1$ indicating that the film/substrate thermal conductivity contrast effect is negligible (Fig. 3). A pair of 30 and $2 \mu \mathrm{~m}$ width heaters was used to measure the temperature difference between the quantum-dots-superlattice sample and a Si substrate reference. For the $30 \mu \mathrm{~m}$ heater width, the spreading effect is very small ( $\beta_{F}$ $\sim 0.066$ ) and according to Fig. 3 the one-dimensional thermal conductivity estimated based on this large width heater would be very close to the true cross-plane thermal conductivity of the film. For the $2 \mu \mathrm{~m}$ heater, $\beta_{F} \sim 0.5$ and Fig. 3 indicates a significant heat spreading effect inside the film and a large error between the estimated one-dimensional thermal conductivity of the film and the true cross-plane thermal conductivity. However, this spreading effect could be used to determine the in-plane thermal conductivity of the film, if a two-dimensional model is employed to fit the temperature drop across the film measured by the small width heater. Therefore, using the pair of above heaters and the two-dimensional heat conduction modeling, both the inplane and cross-plane thermal conductivity of the film could be determined. The experimental data points in Fig. 6 represent the measured temperature rise of the 30 and $2 \mu \mathrm{~m}$ width heaters deposited on the Ge-quantum-dot-superlattice and the reference samples. The experimental signal is compared to predictions for the temperature rise of the heaters calculated based on Eq. (1) for the fitted values of the in-plane and cross-plane thermal conductivity of the film.

## V. EFFECTS OF HEATER HEAT CAPACITANCE AND THERMAL BOUNDARY RESISTANCE

The heater temperature rise given by Eq. (1) was derived by neglecting the heat capacitance of the $3 \omega$ sensor and assuming no thermal boundary resistances present in the system. The purpose of this section is to discuss the possible influence of these effects on the range of applicability of the slope method. The average temperature rise of the heater
![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-09.jpg?height=1156&width=723&top_left_y=165&top_left_x=243)

FIG. 7. Normalized slope [based on the slope from Eq. (20) normalized to $\left.-p / \pi l k_{S}\right]$ compared to the approximate normalized slope of 0.5 is shown as a function of $q b$ : (a) for different $\gamma$ and $R_{\text {th }}=0$, and (b) for $\gamma=1$ and different $\sigma$. The effect of heater thermal capacitance and thermal resistance is to reduce the maximum $q b$ value for which the approximate slope method is applicable, and the effects are more significant at larger $\gamma$ and $\sigma$ values. However, if $\gamma=0$ the thermal resistance does not affect the slope.
element is calculated using a two-dimensional heat conduction model similar to the one outlined in Sec. II, except including a heater of thickness $d_{h}$ and heat capacitance $(\rho c)_{h}$, and a thermal boundary resistance $R_{\text {th }}$ between the heater and the film in contact with the heater. The heat conduction effects inside the heater are neglected. In these conditions, the complex temperature rise of the heater is

$$
\begin{equation*}
T_{h}=\frac{\Delta T+R_{\mathrm{th}} p / 2 b l}{1+(\rho c)_{h} d_{h} i 2 \omega\left(R_{\mathrm{th}}+\Delta T 2 b l / p\right)} \tag{20}
\end{equation*}
$$

where $\Delta T$ is the average complex temperature rise determined from Eq. (1) for a multilayer-film-on-substrate structure under the same heating power and without heater effects. For a heater on a semi-infinite isotropic substrate, Eq. (20) becomes

$$
\begin{equation*}
T_{h}=\frac{p}{2 l k_{S}} \frac{\frac{2}{\pi} \int_{0}^{\infty} \frac{\sin ^{2}(\lambda)}{\lambda^{2} \sqrt{\lambda^{2}+q^{2} b^{2}}} d \lambda+\sigma}{1+\gamma q^{2} b^{2}\left(\frac{2}{\pi} \int_{0}^{\infty} \frac{\sin ^{2}(\lambda)}{\lambda^{2} \sqrt{\lambda^{2}+q^{2} b^{2}}} d \lambda+\sigma\right)} \tag{21}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2025_07_17_f8e4b69b4ee1fc440f3eg-09.jpg?height=543&width=698&top_left_y=176&top_left_x=1174)

FIG. 8. Ratio between the thermal conductivity of the substrate as determined from the slope method for a film-on-substrate system, and the true substrate thermal conductivity, plotted as a function of the film thickness. The calculations are carried out for 2 and $30 \mu \mathrm{~m}$ wide gold heaters of 2000 $\AA$ in thickness, deposited on a $\mathrm{SiO}_{2}$ film on Si -substrate system with different thermal boundary resistances between the heater and the film.

In Eq. (21) the effect of thermal boundary resistance is expressed by the nondimensional parameter $\sigma=R_{\mathrm{th}} k_{S} / b$ and the effect of heater capacitance is indicated by $\gamma$ $=(\rho c)_{h} d_{h} /(\rho c)_{S} b$. Figure 7(a) shows the slope calculated based on Eq. (21), normalized to $-p / \pi l k_{S}$, and plotted as a function of $q b$ parameter for different values of $\gamma$ and $R_{\text {th }}$ $=0$. The range of applicability of the slope method is estimated comparing the normalized slope with the approximate normalized slope value of 0.5 as determined from Eq. (6). The higher is $\gamma$, the larger is the influence of the heat capacitance of the heater, and the smaller is the maximum $q b$ where the approximate slope method is applicable. For example, in order to have less than $1 \%$ relative error in the approximate normalized slope value, if $\gamma=10$ then $q b$ must be smaller than 0.03 [ $\sim 7$ times smaller than predicted by Eq. (9) in the case of no heat capacitance effect]. Furthermore, Fig. 7 shows that if $\gamma=0$ the thermal resistance does not affect the slope. This could be also verified by setting $\gamma=0$ in Eq. (21), then the heater temperature rise could be split in a frequency dependent term (but independent of $R_{\mathrm{th}}$ ) and a frequency independent offset, representing the temperature drop across the thermal resistance. However, if $\gamma \neq 0$ thermal resistance affects the slope, and Fig. 7(b) shows the effect for $\gamma=1$ and different values of $\sigma$. Higher thermal resistances further reduce the maximum $q b$ value where the approximate slope method is applicable.

For a multilayer-film-on-substrate system the additional thermal mass and thermal resistance of the films also affect the applicability of the slope method. To illustrate this effect, Eq. (20) was used to simulate the frequency dependent temperature rise of 2 and $30 \mu \mathrm{~m}$ width gold-heaters $2000 \AA$ thick deposited on a $\mathrm{SiO}_{2}$-film-on-semi-infinite-Si-substrate structure with variable film thickness and thermal boundary resistance between the film and the heater. The approximate slope method employed at current modulation frequencies between 700 and 3000 Hz yields the "measured" substrate thermal conductivity. Figure 8 shows the measured substrate thermal conductivity normalized to the true thermal conductivity of the substrate as a function of film thickness. The
error in the measured substrate thermal conductivity increases with larger film thickness, smaller heater widths, and larger thermal resistance. The trend and magnitude are consistent with our experimental observations and the literature. ${ }^{9}$

## ACKNOWLEDGMENTS

The authors would like to thank Professor K. L. Wang and Professor J. B. Ketterson for the samples used, and the graduate students David W. Song, Diana-Andra Achimov, and Weili Liu for providing the experimental data discussed in this work. This work is supported by the DOD/ONR MURI Grant on Thermoelectrics (N00014-97-1-0516).
${ }^{1}$ D. G. Cahill and R. O. Pohl, Phys. Rev. B 35, 4067 (1987).
${ }^{2}$ D. G. Cahill, Rev. Sci. Instrum. 61, 802 (1990).
${ }^{3}$ S. M. Lee and D. G. Cahill, J. Appl. Phys. 81, 2590 (1997).
${ }^{4}$ S. M. Lee, D. G. Cahill, and R. Venkatasubramanian, Appl. Phys. Lett. 70, 2957 (1997).
${ }^{5}$ G. Chen, S. Q. Zhou, D.-Y. Yao, C. J. Kim, X. Y. Zheng, J. L. Liu, and K. L. Wang, Proceedings of the 17th International Conference Thermoelectrics, Nagoya, Japan, May 24-28 1998, (1998), p. 202.
${ }^{6}$ T. Borca-Tasciuc, D. Song, J. L. Liu, G. Chen, K. L. Wang, X. Sun, M. S.

Dresselhaus, T. Radetic, and R. Gronsky, Mater. Res. Soc. Symp. Proc. 545, 473 (1998).
${ }^{7}$ K. Kurabayashi, M. Touzelbaev, M. Ashegi, Y. S. Ju, and K. E. Goodson, ASME HTD 357-3, 187 (1998).
${ }^{8}$ T. Borca-Tasciuc, J. L. Liu, T. Zeng, W. L. Liu, D. W. Song, C. D. Moore, G. Chen, K. L. Wang, M. S. Goorsky, T. Radetic, and R. Gronsky, ASME HTD 364-3, 117 (1999).
${ }^{9}$ J. H. Kim, A. Feldman, and D. Novotny, J. Appl. Phys. 86, 3959 (1999).
${ }^{10}$ R. Venkatasubramanian, Phys. Rev. B 61, 3091 (2000).
${ }^{11}$ D. W. Song, G. Chen, S. Cho, Y. Kim, and J. B. Ketterson, Mater. Res. Soc. Symp. Proc. 2000 Winter Meeting (in press).
${ }^{12}$ T. Borca-Tasciuc, W. L. Liu, J. L. Liu, T. Zeng, D. W. Song, C. D. Moore, G. Chen, K. L. Wang, M. S. Goorsky, T. Radetic, R. Gronsky, T. Koga, and M. S. Dresselhauss, Superlattices Microstruct. 28, 199 (2000).
${ }^{13}$ T. Borca-Tasciuc, W. L. Liu, J. L. Liu, G. Chen, and K. L. Wang, ASME HTD 366-2, 381 (2000).
${ }^{14}$ A. R. Kumar, D.-A. Achimov, T. Zeng, and G. Chen, ASME HTD 366-2, 393 (2000).
${ }^{15}$ T. Borca-Tasciuc, D. Achimov, W. L. Liu, G. Chen, H.-W. Ren, C.-H. Lin, and S. S. Pei, Proceedings: Heat Transfer and Transport Phenomena in Microsystems, October 15-20, 2000, Banff, Alberta, Canada (2000), p. 369.
${ }^{16}$ H. S. Carslaw and J. C. Jaeger, Conduction of Heat in Solids (Oxford University Press, Oxford, 1959).
${ }^{17}$ D. G. Cahill, S.-M. Lee, and T. Selinder, J. Appl. Phys. 83, 5783 (1998).


[^0]:    ${ }^{\text {a }}$ Current address: Department of Mechanical Engineering, Aeronautical Engineering and Mechanics, Rensselaer Polytechnic Institute, Troy, NY 12180-3590.
    ${ }^{\text {b) }}$ Electronic mail: gchen@seas.ucla.edu

