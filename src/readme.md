warning 
simdのやつは正しく計算できていない。テスト通ってない

しかし、SIMDを強制すると遅くなることが分かったので問題は無視する

以下結果はすべてR9 7959X @4.4GHzによるもの

### 結論
両方縦カラム制において、N=1のとき1threadのdata parallelより２倍程度であるが高速で、
N=manyの時、data parallelと同等の性能を出せる。よし、これでいこう。
N = 1, M = 100000, P = 30000
build ramdom matrix: 1ms97µs500ns
column parallel: 246ms639µs
data parallel: 621ms21µs500ns
both column style jki: 230ms653µs500ns

N = 20000, M = 1000, P = 30000
build ramdom matrix: 93ms347µs
column parallel: 37s388ms501µs300ns
data parallel: 7s235ms666µs500ns
both column style jki: 7s841ms679µs200ns



### わかったこと
・ikjループは８倍程度の高速化をもたらす。
・forの中にスレッド発行を書くと４倍くらい遅くなる。
・ベクトルと行列の積をベクトルの要素ごとに並列してもオーバーヘッドで遅くなる。行列をカラム形式にしても１スレッドと同等にしかならなかった。
一つ目のdata parallelと

N = 1, M = 10000, P = 10000
column parallel: 19ms544µs200ns
data parallel: 18ms22µs600ns

N = 10000, M = 10000, P = 10000
column parallel: 1m11s672ms885µs700ns
data parallel: 12s809ms797µs700ns