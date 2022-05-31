#!/usr/bin/env wolframscript
(* ::Package:: *)

\[Kappa]=1/(Sinh[2k]Sinh[2l]);
l = k;
F[\[Theta]_,k_]=Log[2(Cosh[2 k] Cosh[2 l]+1/\[Kappa](1+\[Kappa]^2-2\[Kappa] Cos[2\[Theta]])^(1/2))];
\[Phi][\[Theta]_,J_,\[Beta]_]=F[\[Theta],k]/.k->\[Beta] J;
f[J_,\[Beta]_]:=-1/(2\[Pi] \[Beta]) NIntegrate[\[Phi][\[Theta],J,\[Beta]],{\[Theta],0,\[Pi]}]
\[Psi][\[Theta]_,J_,\[Beta]_]=D[\[Phi][\[Theta],J,\[Beta]],\[Beta]];
S[J_,\[Beta]_]:=1/(2\[Pi]) NIntegrate[\[Phi][\[Theta],J,\[Beta]],{\[Theta],0,\[Pi]}]-\[Beta]/(2\[Pi])NIntegrate[\[Psi][\[Theta],J,\[Beta]],{\[Theta],0,\[Pi]}];


J = ToExpression[$ScriptCommandLine[[2]]];
Print[S[J,1]]

