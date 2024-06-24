---
baseTemplate: .paperist/templates/index.tex
---

# Deep Learning レポート　第一回
工学系研究科　機械工学専攻　松原貞徳

## (1)Stable Diffusion 
* 概要  
    オープンソースで公開されている生成AIの代表例。適当な言語プロンプトでイラストを簡単に生成できるものとして紹介されており、倫理的な問題が頻繁に議論されている。拡散モデルという技術が使用されている。ここでの技術解説として、[@bishop2024deep]を使用した。
* 実行例(text to image)  
    Input:Crow, Masterpiece, Realistic,  
    Output:  
    ![alt text](../assets/image-1.png)  
    これはローカル環境で実行して生成した画像。
## (2)環境の概要
* 手元のマシンで環境構築を行わない場合  
    https://huggingface.co/spaces/stabilityai/stable-diffusion　
    にアクセスして確かめるのが最も簡単にモデルを試せる。しかし追加学習ができないなど問題点がある。
* ローカルで作業を行いたい場合  
    Githubに公開されているStable-Diffusion-webui-Dockerを用いるのがおそらく最も簡単。Docker DesktopをインストールしたPCでhttps://github.com/AbdBarho/stable-diffusion-webui-docker にアクセスして、案内に従ってコマンドを実行していけばローカルで実行できる環境が構築できる。ただし、GPUがあるパソコンでないと生成が遅い。Readme.mdによるとWindows,LinuxのみでMacはサポートしておらず、AMDのGPUはサポートしていないことに注意。  
    やや面倒だが、追加学習を行う上ではローカルマシンでの環境構築は必須となる。私はゲーミングノートPCで環境構築してみた。スペックは以下の通り  
    cpu:i7-12700h,RAM:64GB,GPU:Nvidia rtx 3070ti

## (3)理論のまとめと不明点
* 拡散モデルの数理的説明  
    BIshopによると拡散モデル(diffusion model)は正式にはデノイズ拡散確率モデル(?)(denoising diffuison probabilistic models,DDPM)と呼ばれる。ガウスノイズを加えノイズあり画像を生成するというプロセスを繰り返し行うことによって、最終的にガウス分布のサンプルに近しい画像が得られるとする。深層学習はこの逆操作を行うために訓練される。すなわちデノイズ後の画像をinput,デノイズ前をoutputとして学習することによって、ガウス分布からのサンプルから始まって適当な出力を画像として得るモデルが得られる。
    ![alt text](../assets/image.png)  
    Gaussian Noiseを加える式は一回目のステップで
    $$
    \boldsymbol{z_1}=  \sqrt{1-\beta_1}+\sqrt{\beta_1}\bm{\epsilon_1}\\
    q(\bm{z_1}|\bm{x})=\mathcal{N}(\bm{z_1}|\sqrt{1-\beta_1}\bm{x},\sqrt{\beta_1}\bm{\epsilon_1})
    $$  
    となり、これを繰り返す。ただし$\epsilon \sim \mathcal{N}(\bm{\epsilon_1}|\bm{0},\bm{I})$とし、確率$q(\bm{z_1}|\bm{x})$に関してはノイズが正規分布に従うという仮定から導かれているということに注意が必要。今欲しいのはこの逆操作、すなわち  
    ![alt text](../assets/image-2.png)
    $$
    p(\bm{z}_{t-1}|\bm{z}_t,\bm{w}) \\
    q(\bm{z}_{t-1}|\bm{z}_t,\bm{w})
    $$
    である。これらはdecoderとして解釈される。
    NNを訓練するための関数として尤度関数が選ばれる。
    $$
    p(\textbf{x}|\textbf{w})=\int \ldots \int p(\textbf{x},\textbf{z}_1,\ldots,\textbf{z}_T|\textbf{w})d\textbf{z}_1 \dots d\textbf{z}_T \\
    =\int \ldots \int p(\textbf{z}_T){\prod_{t=2}^{T}p(\textbf{z}_{t-1}|\textbf{z}_t,\textbf{w})}p(\textbf{x}|\textbf{z}_1,\textbf{w})d\textbf{z}_1 \dots d\textbf{z}_T
    $$
    しかし厳密な推論は手に負えないので、伝統的な方法として変分推論にたよるというアイデアがある。これらはVAEの枠組みでの議論と同様であり、対数尤度の変分下界（evidence lower bound, ELBO）を最大化するというアプローチがとられる。
* 疑問点  
   GPLVMとVAEにはどのような違いがあるのか、そしてその使い分けについて知りたい。NNが特定の条件下でGaussian Processと等価であるというのはRasmussen and Williams(2006)に詳しく、次元圧縮の文脈で主要な手法として紹介されている主成分分析はNNにおいてステップ関数を活性化関数に選んだ2層の自己連想型ネットワークと等価であるとされている。  
   主成分分析にGPを応用して開発されたのがGPLVMであり、VAE(Variational Auto Encoder)はNNをベースにしたAuto Encoderに変分推論を応用したものとして知られている。であれば、VAEとGPLVMには何らかの数学的なつながりが示せそうではある。


