# 现有入口推理测试清单

协议格式仍为 v2，revision 为 existing_entrypoints_only。只使用已有命令，不训练、不改代码或框架；固定 DDPM/100/10 与 H=1/E=1。

models.json 登记 27 项，evaluation_owner 各分 9 项。26 个扩散状态运行现有离线评估；原 token 仅参加闭环。offline_owner 为旧兼容字段，不能据此要求原 token 做扩散离线。

shared/ 的 S0/S1 是共同离线/闭环定义。a/b/c 的第 1 项为各自离线、第 2 项为闭环分片、第 3 项为结果整理，表示分工步骤而非新方法。清单不由现有 CLI 自动读取，不安排新运行器。

完整约定见[团队分工](../../docs/team/README.md)与[测试协议](../../docs/experiments/protocol.md)。当前没有新增正式评测结果。
