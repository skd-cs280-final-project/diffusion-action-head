# 成员 B 动作序列工作区

这里归属连续动作数据窗口、序列头适配、attention padding 和滚动执行。当前生产数据集仍只接受 H=1；这里未提供多步实现。

先完成[成员 B](../../docs/team/member_b.md)的数据连续性审计，再按[接口约定](../../docs/design/interfaces.md)实现。B0 直接引用 A0，B4 复用 B1 权重。保持现有单步路径可用。
