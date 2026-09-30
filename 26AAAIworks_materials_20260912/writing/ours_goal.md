> **历史草稿**：本文件保留早期方案与局部分析；当前引用口径和实验来源见 [写作入口](README.md)。旧数字及推断未经本轮逐项重验，不能直接作为修复后的结论。

我读了PQPP和CATimage，感觉是这样，随着模型基础能力的增强，大多数prompt不是做不做的了，而是做不做好的，或者怎么样更经济的做好，那么之前的事前预测结论是否成立，我们发现在当今模型，对模型在给定prompt下的表现预测是难以精准的，而其之所以可以产生一定的系统调度优势是来自于其对系统以及benchmark的过拟合，而直接预测相对优势可以一定程度上缓解这个问题，我打算这么设计文章与实验 你有什么建议
思考了 3m 27s

我觉得这个方向值得做，而且比单纯再造一个更强的 router 更有研究价值。但你现在的论述需要再收紧，否则容易被审稿人概括成：

“绝对分数回归不准，所以改成预测差值。”

这在方法创新上还不够，而且 RouteT2I 已经直接预测 edge/cloud 两个模型之间的相对质量关系，并据此进行路由；因此“首次预测相对优势”肯定不能再作为贡献。

你真正有机会做成的，是一篇：

对 prompt-only 事前性能预测进行可靠性审计，并将路由目标从绝对性能预测重新表述为边际计算效用估计。

一、先把核心论点改得更精确

你现在有三句话，其中第二句风险最大。

原来的表达
当代模型多数 prompt 已经不是“能不能做”，而是“做得多好、如何经济地做好”。
prompt-level 性能预测难以精准。
现有路由收益来自对系统和 benchmark 的过拟合，而相对优势预测能够缓解。

建议改成：

随着基础模型能力增强，计算分配的关键变量逐渐不再是 prompt 的静态难度，而是该 prompt 从某种额外推理策略中能获得多少边际收益。

Prompt-only 的绝对质量预测受到生成随机性、评价器噪声、模型差异和数据分布变化影响，其 in-domain 可预测性并不等价于可迁移的系统能力。

路由决策本身并不需要准确预测最终质量，而只需要估计不同推理配置之间的增量效用。因此，直接学习 differential utility 可能减少与决策无关的 prompt 难度成分，并提高预算分配的稳定性。

这里最好暂时不要直接用“过拟合”这个词。除非你能通过跨 benchmark、跨模型和模板隔离实验明确证明它，否则审稿人会说：

针对一个确定系统预测其表现，本来就是 system-conditional prediction，为什么叫 overfitting？

更稳妥的词是：

benchmark-conditioned regularities；
system-specific shortcuts；
distribution-specific routing gains；
limited transferability。

等实验做出来后，再决定是否上升到“benchmark overfitting”。

二、你最重要的概念不是“相对优势”，而是“边际效用”

设 prompt 为 x
i
	​

，推理配置为 k，质量为：

μ
k
	​

(x
i
	​

)=E
z
	​

[Q(G
k
	​

(x
i
	​

,z))]

其中 z 是随机种子。

绝对预测学习：

μ
^
	​

k
	​

(x
i
	​

)

你真正需要的是相对于低成本基线 0 的增量：

τ
k
	​

(x
i
	​

)=μ
k
	​

(x
i
	​

)−μ
0
	​

(x
i
	​

)

预算优化可以写成：

{k
i
	​

}
max
	​

i
∑
	​

τ
k
i
	​

	​

(x
i
	​

),s.t.
i
∑
	​

c
k
i
	​

	​

≤B

因为所有 prompt 的基线质量之和 ∑
i
	​

μ
0
	​

(x
i
	​

) 对配置选择是常数。

这带来一个很干净的理论观点：

系统优化并不要求预测 prompt 最终能生成多好，只要求预测额外计算是否值得。

这比“差值回归更容易”更强，因为它说明绝对分数本身包含大量与决策无关的 nuisance component。

例如：

μ
k
	​

(x)=d(x)+r
k
	​

(x)

其中 d(x) 是所有配置共享的 prompt 难度，r
k
	​

(x) 才是配置特异的收益。绝对预测必须同时学习二者，而预算决策只依赖 r
k
	​

−r
0
	​

。

三、第一张图应该直接推翻“困难 prompt 应该获得更多算力”

建议横轴画低成本配置质量，例如 10-step score：

x=μ
10
	​

(p)

纵轴画最优额外计算收益：

y=max{μ
40
	​

(p),μ
TTS
	​

(p)}−μ
10
	​

(p)

由此得到四类 prompt：

低成本表现	额外计算收益	含义
高	低	已经做好，继续算是 over-computation
低	高	真正应该被 under-serving 的样本
低	低	当前推理策略无法挽救，继续算也没用
高	高	已经不错，但仍有明显提升空间

最重要的是证明：

Corr(μ
10
	​

,μ
40
	​

−μ
10
	​

)

很低，甚至不同 benchmark 下方向不一致。

这样你攻击的就不是某一个 predictor，而是更基础的假设：

Prompt difficulty is not compute utility.

CATImage 的主要直觉仍是把昂贵配置留给“复杂 prompt”，但其论文自己也给出了计数 prompt 上增加 denoising steps 反而降低质量的失败案例，并承认随机噪声带来的不确定性没有被显式处理。

这正好能成为你的切入点。

四、实验要分成四个层次
1. 先测“这个东西理论上有多可预测”

不能只报告模型的 Pearson 或分类准确率。需要先估计标签本身的可靠性。

对每个 prompt/configuration 使用配对随机种子，最好：

全量至少 4 个 seed；
选一个较大子集跑 8–16 个 seed；
同模型不同 step 尽量使用相同初始噪声。

然后计算：

seed split-half correlation；
configuration difference 的置信区间；
absolute score 和 delta score 的信噪比；
prompt × configuration interaction variance；
不同 scorer 之间的 route agreement。

可以用混合效应分解：

Y
i,k,z,j
	​

=μ+P
i
	​

+C
k
	​

+(P×C)
i,k
	​

+S
z
	​

+J
j
	​

+ϵ

其中真正可以用于路由的是 P×C，而不是总体 prompt 难度 P
i
	​

。

如果 prompt×configuration variance 小于 seed/judge variance，那么你就可以非常有力地说：

prompt-only 路由受到标签可识别性的根本限制，而不只是 predictor 不够强。

PQPP 已经发现，跨生成模型测试会出现明显的性能下降，而且基于 HPSv2 的自动标签比人工标签更容易预测。 你的工作需要进一步把这种“不可靠性”连接到实际预算决策，而不只是相关系数。

2. 公平比较不同预测目标

固定完全相同的 Qwen embedding、MLP 容量和训练数据，只改变监督目标：

每个配置的绝对分数回归；
baseline + delta；
直接连续 delta 回归；
pairwise ranking；
best-configuration classification；
delta regression + ranking joint loss。

这里有一个关键点：

不要只预测谁赢。

在全局预算有限时，知道 40-step 是否比 10-step 好还不够，还要知道：

提升是 0.01 还是 0.20；
哪些 prompt 最值得优先升级。

因此主模型应该预测连续 incremental utility，而 pairwise classification 只能作为辅助 loss 或 baseline。

建议预测：

τ
^
40
	​

(x),
τ
^
TTS
	​

(x)

并额外输出不确定性：

σ
^
40
	​

(x),
σ
^
TTS
	​

(x)

优化时可使用保守效用：

τ
~
k
	​

=
τ
^
k
	​

−β
σ
^
k
	​


这比单纯的差值 MLP 更像一个完整方法。

3. 真正证明是否存在 benchmark/system shortcut

至少需要以下测试矩阵：

训练	测试	检验内容
Geneval2	Geneval2 IID	标准 in-domain
Geneval2	CoreBench / WISE	跨 benchmark
CoreBench	Geneval2	反向迁移
Qwen	Flux	跨模型
自动 scorer A	scorer B / 人评	跨评价目标
原始模板	paraphrase / held-out template	模板记忆
随机 split	semantic-cluster split	近邻泄漏

再增加几个很有解释力的弱基线：

benchmark category ID；
prompt 长度、名词数、数字数；
category mean utility；
KNN；
bag-of-words；
frozen embedding + linear head。

如果 category-only 或 KNN 在 IID 上接近复杂 router，但在 leave-one-category-out 时一起崩溃，你才有资格说当前 routing gain 很大程度上来自 benchmark-specific regularities。

另外，训练集和测试集一定要做 embedding-cluster split。随机划分 COCO caption、模板化 benchmark 时，语义近邻泄漏会非常严重。CATImage 本身主要在每个数据集内部独立进行 80/10/10 划分，因此它证明的是 in-distribution routing effectiveness，不足以说明跨分布稳健性。

4. 最终评价必须以决策为中心

预测指标报告：

R
2
；
Pearson / Kendall；
calibration；
sign accuracy；
top-k gain recall。

但主要结果必须是：

fixed-cost quality；
quality–cost Pareto AUC；
regret to oracle；
oracle-over-random gap recovery；
harmful upgrade rate；
budget violation；
route stability across seeds/judges。

尤其推荐：

Oracle Gap Recovery=
U
oracle
	​

−U
random
	​

U
router
	​

−U
random
	​

	​


因为它能直接显示：即使绝对预测相关性一般，路由究竟捕捉了多少真正可利用的 signal。

五、必须正面处理 RouteT2I，而不能只谈 PQPP 和 CATImage

RouteT2I 已经：

直接预测 edge/cloud 输出之间的 Pareto relative superiority；
只把云模型明显占优的 prompt 路由到云端；
在多个模型 pair 上测试相对路由。

你的区别应该明确写成：

RouteT2I 研究的是两个模型的 pairwise edge-cloud routing；你研究的是三个及以上、成本不同且机制不同的 inference policies。
它主要证明相对路由有效；你研究绝对预测和相对预测的可识别性、迁移性和决策可靠性。
它使用固定 cloud routing rate；你处理严格的全局共享预算和多档配置分配。
它定义自己的多维 PRS；你测试 benchmark scorer、通用 reward model 和人评之间的 judge shift。
你的核心不是提出另一个 pairwise router，而是解释：为什么 in-domain absolute prediction 看似有效，却可能无法支持 robust deployment。
六、关于“现代模型能力增强”的论点，需要一个小型纵向实验

这句话很有感染力，但目前还是 intuition。建议加入一组旧模型与新模型对照，不需要做得很大：

较旧基础模型：SDXL 或其他容易复现的模型；
当前模型：你的 Qwen 和 Flux；
同一批 prompt；
比较低成本配置的失败率、分数方差和额外计算收益分布。

你想看到的是：

新模型低成本表现整体上升；
明确失败 prompt 的比例下降；
不同配置之间的平均差距变小；
但 prompt-specific incremental gain 仍然存在；
incremental gain 相对于 seed noise 更难预测。

这才能支撑：

随着模型能力提高，路由问题从 failure avoidance 转向 fine-grained utility allocation。

否则建议在 introduction 中写成“our observations on contemporary models”，不要写成普遍历史规律。

七、我建议的论文定位

最好的定位不是：

我们提出一种比 CATImage 更强的路由方法。

而是：

现有工作把 prompt performance prediction 当作计算路由的可靠基础，但我们发现，绝对质量预测的 in-domain 成功会掩盖其有限的可识别性和迁移性。预算分配真正需要的不是预测最终质量，而是估计 inference policy 的增量效用。

可以凝练成三项贡献：

Empirical diagnosis：证明静态 prompt difficulty 与额外计算收益并不一致，并量化 prompt-only 性能预测的 seed、judge、model 和 benchmark 不确定性。
Decision-centric formulation：把预算化 T2I 推理重写为 incremental utility estimation，而非 absolute quality prediction。
Robust evaluation and method：提出直接 delta prediction、ranking 和 uncertainty-aware allocation，并在 IID 与多种 distribution shift 下比较。

我最推荐的标题是：

From Prompt Difficulty to Compute Utility: Rethinking Performance Prediction for Budgeted Text-to-Image Inference

更有观点的版本是：

Do We Need to Predict Image Quality? A Reliability Study of Prompt-Based Routing for Text-to-Image Generation

或者偏方法：

Predicting Less, Routing Better: Differential Utility Estimation for Budgeted Text-to-Image Generation

一句话评价：**这个故事有潜力，但必须从“差值预测的小改进”升级成“事前性能预测是否可靠，以及路由究竟应该预测什么”的系统性研究。**这才足以和 PQPP、CATImage、RouteT2I 拉开层次。

Sources
你理解错我的文章基调了，我不是方法文章，我是要说 他们的这个东西 在现在的模型基础能力上 不work了，我要指出绝对能力预测就学不好，并提出缓解的小方案 主要是一个分析 唱反调文章
思考了 48s

对，这样我就理解了。你的文章不是：

我们提出一个更好的计算分配方法。

而是：

Prompt-only 的事前绝对性能预测，在旧模型和弱模型时代看起来合理，但在当前强文生图模型上已经变成一个低可预测性、低可迁移性的目标。现有工作报告的系统收益，并不能证明它真正学会了模型对单个 prompt 的生成能力。

然后，“直接预测相对优势”只是用来说明：

问题不一定是完全没有信号，而是绝对能力预测这个任务定义包含了大量与系统决策无关、且难以预测的因素。

这个基调确实和我刚才说的“做一个更完整的 utility routing 方法”不同。你应该把文章写成一篇 empirical challenge / negative-result analysis paper，而不是方法论文。

一、真正要挑战的命题是什么

你挑战的不是“动态计算分配有用吗”，而是下面这个被默认接受的前提：

给定一个 prompt，可以仅根据 prompt，在生成前可靠预测某个文生图模型最终会表现得多好。

PQPP 明确把任务定义为 pre-generation prompt performance prediction；CATImage 则进一步把 prompt complexity 或预测质量用于选择生成模型和去噪步数。PQPP 的实验包括 SDXL 与 GLIDE，CATImage 主要在 COCO、DiffusionDB 上学习对多个已有生成函数进行路由。

你的唱反调观点可以是：

The apparent success of pre-generation performance prediction does not necessarily indicate that a predictor understands a model’s prompt-level capability. On contemporary text-to-image models, absolute generation quality is weakly identifiable from the prompt alone, and much of the reported predictive or routing utility may come from benchmark- and system-specific regularities.

中文可以概括为：

事前性能预测可能学到的是“这个 benchmark 上哪些语言模式通常得分高”，而不是“这个模型面对这个 prompt 会生成得怎么样”。

这就是整篇文章的中心。

二、“在当前模型上不 work”一定要被严格定义

不能只用一个较低的 Pearson，然后说“不 work”。审稿人会马上反驳：

相关系数虽低，路由收益不是仍然存在吗？
是不是你的预测器不够强？
是不是数据量不够？
是不是 evaluator 太噪？
是不是你选择的 benchmark 有问题？
随机生成本来就不可确定，预测期望不就行了吗？

因此你要把“不 work”拆成四个可以验证的命题。

命题一：绝对质量本身难以从 prompt 识别

固定模型 m，对于 prompt x，你要预测的是：

q
m
	​

(x)=E
z
	​

[S(G
m
	​

(x,z),x)]

但现代模型的观测分数可能更像：

Y
x,z,j
	​

=μ+P
x
	​

+Z
x,z
	​

+J
x,j
	​

+ϵ

其中：

P
x
	​

：prompt 稳定决定的难度；
Z
x,z
	​

：随机种子带来的变化；
J
x,j
	​

：评价器或评价维度带来的变化；
ϵ：剩余噪声。

你需要证明，在强模型上：

Var(P
x
	​

)

相对于随机种子、评价器和图像实例导致的方差已经很小。

也就是说，不是 predictor 不够强，而是prompt 可提供的稳定预测信号变弱了。

命题二：模型越强，绝对质量越难预测

这才直接支撑你“时代变化”的基调。

旧模型时代，prompt 之间可能存在非常明显的差异：

“a dog”容易；
十个物体、复杂空间关系、文字渲染很难；
简单 prompt 和复杂 prompt 的质量差距很大。

强模型时代，大多数 prompt 都能生成一个“还行”的结果，剩余差异变成：

数量差一个；
某个空间关系错；
某一随机种子失败；
evaluator 对细节的判断不稳定；
不同模型擅长点不同。

因此绝对性能的动态范围被压缩，难度不再是粗粒度的“做不做得到”，而是细粒度的“这一次具体做得有多好”。

你需要直接画一张趋势图：

横轴：模型整体基础能力；
纵轴：prompt-only absolute prediction 的 R
2
、Kendall 或 noise-normalized predictability；
每个点是一代模型或一个能力档位。

预期结果是：

模型平均质量提高，但 absolute-score predictability 下降。

这会是文章最重要的经验发现。

命题三：IID 预测结果高估了真实泛化能力

你要证明随机划分下得到的预测能力，部分来自：

benchmark 模板；
prompt 类别；
高频语言模式；
语义近邻；
特定模型的系统性偏好；
特定 scorer 的打分规律。

因此至少需要比较：

random split；
semantic-cluster split；
held-out category split；
held-out template split；
cross-benchmark；
cross-model；
cross-evaluator。

一旦 random split 尚可，而 cluster、benchmark、model 或 evaluator shift 下明显崩溃，你就能说：

现有结果更接近 distribution-specific score prediction，而不是 transferable capability prediction。

命题四：系统收益不等于绝对性能预测成功

这是你文章里最有意思的一层。

即使绝对预测相关性很差，一个 router 仍可能提高总分，因为它只需要在少量关键样本上做对，或者利用 benchmark 中稳定的类别先验。

例如：

counting prompt 更可能走 TTS；
简单单物体 prompt 更可能走 10 steps；
某类空间 prompt 在 Qwen 上更容易受益；
某些 benchmark 模板总是偏向某种配置。

这些规则可能足以产生系统收益，但它不代表 predictor 能准确预测：

q
10
	​

(x),q
40
	​

(x),q
TTS
	​

(x)

所以你的结论应该是：

Routing utility is not evidence of accurate prompt-level performance prediction.

这句话很适合放在摘要和 introduction 末尾。

三、整篇文章最自然的逻辑
第一幕：一个曾经合理的假设

事前性能预测从信息检索的 QPP 延伸到文生图：prompt 中包含难度信息，因此可以在生成前预测结果质量。PQPP 正式提出并评测这一任务；CATImage 则进一步把类似预测用于计算与模型路由。

这个逻辑在较弱模型上很自然：

简单 prompt 做得好，复杂 prompt 做得差。

第二幕：模型能力增强改变了问题

当前模型已经可以为多数 prompt 生成语义合理的图片。问题逐渐由：

能不能完成？

变成：

哪一种推理配置能更经济地完成得更好？

在这个新阶段，绝对质量受到大量图像实例和 evaluator 因素影响，prompt 本身不再足以确定最终表现。

第三幕：系统性审计发现绝对预测不可靠

你的实验发现：

强模型的 absolute-score predictor 学不好；
增加 predictor 容量或数据量不能根本解决；
标签自身的跨 seed、跨 evaluator 稳定性有限；
random-split 表现不能迁移；
简单 benchmark/category prior 已经可以解释一部分路由收益。

因此，现有范式把两个问题混在一起了：

Capability estimation：预测模型对 prompt 的绝对表现；
System optimization：选择相对更合适的推理配置。

前者可能失败，后者仍然可能局部有效。

第四幕：相对预测作为诊断性缓解

你不是说“我们终于解决了路由”，而是：

如果把目标从绝对分数改成同一 prompt 在不同配置之间的相对收益，一部分共同噪声和无关难度可以被抵消。

例如：

Δ
40
	​

(x)=q
40
	​

(x)−q
10
	​

(x)
Δ
TTS
	​

(x)=q
TTS
	​

(x)−q
10
	​

(x)

直接预测 Δ 比先预测三个绝对值再相减更合理。

但你的措辞应当保持克制：

Relative prediction partially mitigates the problem but does not eliminate its limited transferability.

这才符合唱反调分析文章的气质。

四、实验不要围绕“我的方法比别人高多少”

实验应该围绕一连串诊断问题。

RQ1：绝对性能在现代模型上究竟有多可预测？

固定 embedding 和 predictor，测试：

linear regression；
random forest；
MLP；
fine-tuned text encoder；
LLM-based regression。

目的不是刷模型，而是排除“预测器太弱”。

指标：

R
2
；
Pearson；
Kendall；
MAE；
ranking accuracy；
calibration。

其中最重要的是 R
2
。因为某些情况下 Pearson 看起来还行，但实际只能解释极少的质量方差。

同时画：

predicted vs. ground-truth scatter；
score histogram；
learning curve；
predictor capacity curve。

如果训练集上可以拟合、测试集上不行，说明泛化问题；如果训练集都难以拟合，说明 prompt 表征或标签本身信息不足。

RQ2：是 predictor 不好，还是标签根本不稳定？

每个 prompt/configuration 跑多个 seed，至少：

全量 4 seeds；
代表性子集 8–16 seeds。

然后计算：

单 seed 与多 seed 均值的相关性；
两组 seed 均值之间的 split-half correlation；
prompt 排名在不同 seed 组下的一致性；
随 seed 数增加，预测上限如何变化；
绝对分数置信区间。

最好估计一个 empirical noise ceiling。

例如，使用一半 seed 得到的均值预测另一半 seed 的均值。如果 ground-truth 自己都只能达到相关系数 0.4，那么 predictor 达到 0.3 不能简单理解为模型不够强。

这一步会让你的 negative result 非常扎实。

RQ3：模型变强后，可预测性是否真的下降？

这一步是整篇文章是否成立的关键，不能省。

可以选择：

旧模型；
中等模型；
当前强模型。

或者同一模型的不同能力设置：

极低步数；
正常步数；
更强推理策略。

但最好还是有跨代模型，因为“10 steps 和 40 steps”不完全等价于基础能力演进。

每个模型报告：

mean quality；
prompt-level quality variance；
seed variance；
absolute prediction R
2
；
noise ceiling；
normalized predictability。

一个很有力的指标是：

Normalized Predictability=
R
noise ceiling
2
	​

R
predictor
2
	​

	​


否则强模型分数范围缩小会自然导致相关性下降，审稿人会说只是 range restriction。你必须区分：

绝对方差缩小；
标签变噪；
可提取的 prompt signal 确实下降。
RQ4：现有预测器究竟学到了什么？

这是“benchmark/system overfitting”部分。

做以下 baselines：

prompt 长度；
token 数；
数字数量；
实体数量；
benchmark category；
template ID；
category mean score；
nearest-neighbor score；
bag-of-words linear model；
frozen embedding linear model。

然后做几个很杀人的实验。

类别内标签打乱

在每个 benchmark category 内打乱具体 prompt 的分数，但保留类别均值。

如果 router 仍然能获得相当比例的系统收益，说明它主要利用类别先验，而不是实例级能力预测。

Leave-one-category-out

训练时完全不见某类能力，例如 counting 或 spatial，测试时只测试该类。

若性能显著下降，说明 predictor 没有学到一般性的 prompt difficulty。

Paraphrase consistency

对同一个语义 prompt 做多个改写。理论上它们对应的能力和配置收益应接近。

检查：

Var(
q
^
	​

(x
(1)
),…,
q
^
	​

(x
(k)
))

如果轻微措辞变化导致预测明显变化，而实际生成表现变化不大，就能暴露 surface shortcut。

跨 benchmark 和跨模型
Geneval2 → CoreBench；
CoreBench → WISE；
Qwen → Flux；
Flux → Qwen。

PQPP 本身已经报告了 SDXL 与 GLIDE 之间的 cross-model prediction setting，因此你的文章最好把这件事扩展到现代模型，并进一步分析下降的来源，而不是仅报告一次跨模型结果。

RQ5：为什么绝对预测不准，路由仍能取得收益？

这一组实验应该直接拆解系统 gain。

比较：

oracle routing；
random routing；
absolute-score prediction routing；
category-prior routing；
nearest-neighbor routing；
direct relative prediction；
fixed configuration。

然后报告：

absolute prediction accuracy；
routing utility；
oracle gap recovery；
cross-distribution routing utility。

你可能会得到一个非常有价值的结果：

方法	绝对预测	IID 路由	OOD 路由
Absolute predictor	较差	尚可	很差
Category prior	无	尚可	很差
Relative predictor	不预测绝对值	更好	略好
Oracle	—	高	高

这张表能清楚说明：

IID routing improvement can coexist with poor capability prediction.

RQ6：相对预测能缓解多少？

这里保持简单就行，别做成方法 section。

只比较三种方案：

预测每个配置的绝对分数，再相减；
直接预测连续相对收益；
直接预测 pairwise winner。

你的预期结论不是“我们 SOTA”，而是：

直接预测相对收益通常比绝对预测后作差更稳定；
在跨 seed、跨 benchmark 或跨模型时下降更缓；
但仍然没有达到可称为通用能力预测的水平。

这就是一个很好的 constructive takeaway：

When routing is needed, predict the decision-relevant contrast rather than the absolute outcome—but do not mistake this for reliable capability estimation.

另外，RouteT2I 已经使用了两个模型之间的相对优越性进行 edge-cloud routing，所以你不要把 relative prediction 包装成新方法。它可以作为一个已有方向支持你的分析：任务真正需要的可能本来就不是绝对质量。

五、你需要特别避免的三个坑
1. 不要直接说“多数 prompt 都做得好了”

这句话过强，而且审稿人会拿 compositionality、counting、text rendering、复杂空间关系反驳你。

更稳妥的是：

On contemporary models, failures increasingly shift from coarse prompt-level infeasibility to fine-grained, sample-dependent errors.

中文：

当前模型并没有解决所有 prompt，但错误形态正从稳定的、粗粒度的 prompt 难度，转向更细粒度、更依赖具体采样结果的失败。

这比“现在什么都能做”准确得多。

2. 不要一开始就把问题归因于 overfitting

“过拟合”必须是实验结论，而不是前提。

建议逐级用词：

in-domain regularities；
benchmark-specific shortcuts；
system-conditioned correlations；
limited cross-distribution validity；
最后有充分证据再说 overfitting。

特别是“对系统过拟合”有概念风险。预测特定系统的性能本来就应当依赖该系统。你真正批评的是：

这些 predictor 被呈现为 prompt difficulty 或 capability prediction，但实际上只是特定 model–benchmark–evaluator tuple 上的条件预测器。

这个说法更精准。

3. 不要因为绝对预测失败，就声称动态路由不成立

你的实验已经说明动态分配可以有效。硬说 routing 完全没用，会和自己的结果冲突。

你的观点应该是：

Routing may work, but not for the reason commonly implied.

或者更直接：

The system works; the capability-prediction interpretation does not.

我觉得这甚至可以作为论文里的醒目句子。

六、文章的贡献应该怎么写

建议控制为三点。

Contribution 1：范式审计

We revisit pre-generation prompt performance prediction on contemporary text-to-image models and find that absolute prompt-level quality is only weakly predictable from the prompt alone.

Contribution 2：失败机制

Through multi-seed variance decomposition and extensive distribution shifts, we show that the apparent in-domain success is partly explained by benchmark-, evaluator-, and system-specific regularities rather than transferable prompt-level capability estimation.

Contribution 3：建设性结论

We disentangle capability prediction from routing utility and show that directly predicting relative gains partially recovers decision-relevant signals, while remaining far from a general solution.

注意第三点是 takeaway，不是“我们提出了一个新 router”。

七、我会怎么排文章结构
1. Introduction

提出旧范式和新模型能力之间的错位：

以前预测“这个 prompt 难不难”有意义；
当前模型的失败更细粒度、更随机；
我们质疑 prompt-only absolute performance prediction 是否仍然是合理任务；
主要发现：absolute prediction weak、IID 假象、routing ≠ capability estimation、relative target 部分缓解。
2. Revisiting Pre-generation Performance Prediction

定义：

absolute performance prediction；
relative performance prediction；
routing utility；
三者之间的区别。

这部分不要放太多方法。

3. Experimental Audit Setup

说明：

多模型；
多 benchmark；
多 evaluator；
多 seed；
多 predictor；
IID/OOD splits；
可靠性与 noise ceiling。
4. Does Absolute Performance Prediction Still Work?

主结果：

不同模型代际；
不同 predictor；
learning curve；
方差分解；
noise ceiling。
5. What Do Prompt Predictors Actually Learn?
benchmark shortcuts；
category/template；
semantic leakage；
paraphrase；
cross-model/evaluator/benchmark。
6. Why Can Routing Still Work?
capability prediction 与 routing utility 解耦；
category prior；
oracle gap；
IID gain 与 OOD collapse。
7. A Modest Mitigation: Predicting Relative Gains

很克制地放一个小节：

absolute-to-delta；
direct delta；
pairwise；
改善但未解决。
8. Discussion

讨论：

prompt-only 预测的适用边界；
事前预测可能需要结合 early generation signals；
benchmark routing 不能等同部署能力；
未来应强调 uncertainty、post-generation feedback 或在线观测。
八、最适合你的标题

我最推荐：

Does Pre-generation Performance Prediction Still Work for Modern Text-to-Image Models?

它直接、明确，很像分析论文。

更尖锐的：

The Illusion of Prompt-Level Performance Prediction in Modern Text-to-Image Generation

但 “Illusion” 需要实验非常硬，否则显得攻击性过强。

更学术、稳健的：

Rethinking Prompt Performance Prediction for Contemporary Text-to-Image Models

突出核心发现的：

When Prompt Difficulty Stops Predicting Generation Quality

强调系统收益与预测能力错位的：

Routing Works, but Performance Prediction Does Not

这个很抓人，但也最需要你把“does not”定义得严谨。

九、你这篇文章最终最有价值的结论

不是：

relative predictor 比 absolute predictor 高几个点。

而应该是下面这组结论：

现代文生图模型并没有消除困难 prompt，但它们改变了失败的统计形态。模型表现越来越难被归结为一个由 prompt 单独决定的稳定绝对能力值。

因此，随机划分下的预测相关性和路由收益，不足以证明 predictor 学会了模型的 prompt-level capability；它可能只捕捉了特定 benchmark、模型和评价器中的条件规律。

相对收益比绝对质量更接近系统实际需要的信号，但其作用是缓解任务错配，而不是解决通用能力预测。

这样文章就是一篇真正有立场的分析工作：不是否定所有路由，而是否定“通过 prompt 事前预测绝对能力，是理解并调度现代生成模型的可靠基础”这一默认叙事。