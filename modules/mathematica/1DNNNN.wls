#!/usr/bin/env wolframscript
(* ::Package:: *)

TW4N=Partition[Flatten[
Table[KroneckerDelta[s2,s2p]KroneckerDelta[s3,s3p]Exp[k1 s1 s2 + k2 s1 s3+k3 s1 s4],{s1,{1,-1}},{s2,{1,-1}},{s3,{1,-1}},{s2p,{1,-1}},{s3p,{1,-1}},{s4,{1,-1}}]
],8];
lambdaTW4N=Eigenvalues[TW4N];
f4=-1/\[Beta] Log[lambdaTW4N[[2]]]/.{k1->\[Beta] k1, k2->\[Beta] k2, k3->\[Beta] k3};
S4=\[Beta]^2 D[f4,\[Beta]]/.\[Beta]->1 //Simplify;


k1 = ToExpression[$ScriptCommandLine[[2]]]
k2 = ToExpression[$ScriptCommandLine[[3]]]
k3 = ToExpression[$ScriptCommandLine[[4]]]
Print[S4]



