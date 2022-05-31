#!/usr/bin/env wolframscript
(* ::Package:: *)

S[k1_,k2_]:=1/2 (-((2 E^(2 k2) (-1+E^(2 k1)) k1)/Sqrt[4 E^(2 k1)+E^(4 k2) (-1+E^(2 k1))^2])+(8 E^(k1+2 k2) k2 Cosh[k1])/((-1+E^(4 k2)) Sqrt[4 E^(2 k1)+E^(4 k2) (-1+E^(2 k1))^2])-2 k2 Coth[2 k2]-Log[2]+Log[E^(-2 (k1+k2)) (2 E^(2 k1)+E^(4 k2)+E^(4 (k1+k2))+E^(2 k2) (1+E^(2 k1)) Sqrt[4 E^(2 k1)+E^(4 k2) (-1+E^(2 k1))^2])])


k1 = ToExpression[$ScriptCommandLine[[2]]]
k2 = ToExpression[$ScriptCommandLine[[3]]]
Print[S[k1,k2]]



