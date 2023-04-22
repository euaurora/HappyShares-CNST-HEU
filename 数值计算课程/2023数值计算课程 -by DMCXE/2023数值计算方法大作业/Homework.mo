model Homework
parameter Real G1 =   0.0035;
parameter Real G2 =   0.131;
parameter Real G3 =   0.028;
parameter Real tau0 = 4.7;
parameter Real tau =  6.6;
parameter Real xi =   1.68;
parameter Real beta = 0.2;
Real qe;
Real qv;
Real x1;
Real x2;
Real x3;
Real x4;
Real y;

function qvt
  input Real t;
  output Real qv;
  algorithm
    if t < 5 then
    qv := 0;
  else
    qv:= 1;
  end if;
end qvt;

function qet
  input Real t;
  output Real qe;
  algorithm
    if t < 20 then
    qe := 0;
  else
    qe:= 1;
  end if;
end qet;


initial equation
x1 = 0;
x2 = 0;
x3 = 0;
x4 = 0;

equation
qv = qvt(time);
qe = qet(time);
der(x1) = G1 * (qe-qv);
der(x2) = -x2/tau0 + G2*qv/tau0;
der(x3) = x4/tau^2 - G3*beta*qe/tau;
der(x4) = -x3 - 2*xi*x4/tau + (2*xi*beta - 1)*G3*qe;
y = x1 + x2 + x3;
end Homework;