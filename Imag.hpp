#ifndef IMAG_HPP
#define IMAG_HPP
/*! \file Imag.hpp
    \brief Real and Complex numbers
 
    A small definition of real and complex number with some operation. Both real and complex can be in single or double precision. Default is double, tu use single precision the flag -DSINGLE need to be use at compilation.
 
 */ 

#ifdef SINGLE
typedef float Real; /*! \typedef Real
                        \brief real numbers
                     */
#else
typedef double Real;
#endif

/*! \class Imag
    \brief complex numbers
 
    Complex number, defined as Real[2] if FFT capability of latfield are not used, and with FFTW complex if it is use. Commun operation over complex number are also defined.
 
 */

class Imag
{



 private:
	
#ifndef WITH_FFTS
	Real data[2];
#endif
    
#ifdef WITH_FFTS   
#ifdef SINGLE
	fftwf_complex data;
#endif
#ifndef SINGLE
	fftw_complex data;
#endif	
#endif
       
 public:
  //CONSTRUCTORS
  Imag() {;};
  Imag(Real a,Real b) { data[0]=a; data[1]=b; };

  //NEGATION OPERATOR
  Imag operator-() { return Imag(-data[0],-data[1]); }
  
  //IMAGINARY-IMAGINARY ADDITION, ETC OPERATORS
  Imag operator+(Imag z) { return Imag( data[0]+z.real(), data[1]+z.imag() ); }
  Imag operator-(Imag z) { return Imag( data[0]-z.real(), data[1]-z.imag() ); }
  Imag operator*(Imag z) { return Imag( data[0]*z.real()-data[1]*z.imag(), data[0]*z.imag()+data[1]*z.real() ); }
  Imag operator/(Imag z) { return Imag( (data[0]*z.real()+data[1]*z.imag())/z.norm(), (data[1]*z.real()-data[0]*z.imag())/z.norm() ); }

  void operator=(Real r) {data[0]=r;data[1]=0;}



  //Real-IMAGINARY ADDITION, ETC OPERATORS
  friend Imag operator+(Imag z,Real a) { return Imag(z.real()+a, z.imag()); }
  friend Imag operator+(Real a,Imag z) { return Imag(z.real()+a, z.imag()); }
  friend Imag operator-(Imag z,Real a) { return Imag(z.real()-a, z.imag()); }
  friend Imag operator-(Real a,Imag z) { return Imag(a - z.real(), z.imag()); }
  friend Imag operator*(Imag z,Real a) { return Imag(z.real()*a, z.imag()*a); }
  friend Imag operator*(Real a,Imag z) { return Imag(z.real()*a, z.imag()*a); }
  friend Imag operator/(Imag z,Real a) { return Imag(z.real()/a, z.imag()/a); }
  friend Imag operator/(Real a,Imag z) { return Imag(a,0.0)/z; }

  //SELF ADDITION, ETC WITH Imag OPERATORS
  void operator+=(Imag z) { data[0] += z.real(); data[1] += z.imag(); }
  void operator-=(Imag z) { data[0] -= z.real(); data[1] -= z.imag(); }
  void operator*=(Imag z) { Real temp = data[0]; data[0] = temp*z.real() - data[1]*z.imag(); data[1] = temp*z.imag() + data[1]*z.real(); }

  //SELF ADDITION, ETC WITH Real OPERATORS
  void operator+=(Real a) {data[0] += a; }
  void operator-=(Real a) {data[0] -= a; }
  void operator*=(Real a) {data[0] *= a; data[1] *= a; }
  void operator/=(Real a) {data[0] /= a; data[1] /= a; }

  //COMPLEX NUMBER FUNCTIONS
  Real& real() { return data[0]; }
  Real& imag() { return data[1]; }
  Real phase() { return acos( data[0]/sqrt(data[0]*data[0]+data[1]*data[1]) ) * ( data[1] < 0 ? -1 : 1 ); }
  Imag  conj() { return Imag(data[0],-data[1]); }
  Real  norm() { return data[0]*data[0] + data[1]*data[1]; }


  //TRIGONOMETRIC FUNCTIONS
  friend Imag sin(Imag z) { return Imag( std::sin(z.real())*cosh(z.imag()), std::cos(z.real())*sinh(z.imag()) ); }
  friend Imag cos(Imag z) { return Imag( std::cos(z.real())*cosh(z.imag()), -std::sin(z.real())*sinh(z.imag()) ); }
  friend Imag expi(Real x) { return Imag( std::cos(x), std::sin(x) ); }

  //I/O STREAM OPERATORS
  friend std::ostream& operator<<(ostream& os, Imag z) { os<<z.data[0]<<" "<<z.data[1]; return os; }
  friend std::istream& operator>>(istream& is, Imag& z) { is>>z.data[0]>>z.data[1]; return is; }
};



#endif
