# RDK X5 侧技术路线与程序规划（网页版fable5，将转向codex开发）

**项目**：越野轮足双足机器人 RL 部署（X5 上位机侧）
**版本**：v0.4（2026-07-17）· v0.2 合并另一版技术路线（附录 D）；v0.3 审视官方 LeRobot 生态，核心架构不重构（附录 E）；
        v0.4 定稿遥控方案（**2.4GHz 无线手柄直连 X5**），删除机械臂，并按官方《快速开始》(v3.5.0) 补全系统基线。
**建议用法**：放入仓库 `docs/PLAN.md`，仓库根目录 `CLAUDE.md` 摘要引用（见 §12）。

---

## 0. 开工前确认准备（确认后勾选，Claude Code 依赖这些事实）

- [ ] **电机顺序 MotorIdx** 与 H723 固件 CAN ID 的对应关系（§4.5 为占位）
- [ ] **轮足动作语义**：速度模式 —— **以 Isaac Lab env 的 actuator 配置为准**（见 §4.6；训练与部署必须同一种）
- [ ] **Isaac Lab env 的 observation / action 配置原文**（关节顺序、obs_scales、action_scale、clip、default_pos、decimation×dt、commands 维度）→ 经 `tools/export_cfg.py` 生成 `config/policy.yaml`
- [ ] 各腿关节 **kp/kd、零位偏置、软限位、力矩限幅**；轮子限幅 → `config/robot.yaml`
- [ ] 机器人 **URDF / MJCF**（sim2sim 用；从训练资产转换）
- [ ] H723 USB 的 **VID/PID/序列号**（udev 固定设备名，§11.2）
- [x] **遥控**：2.4GHz 无线手柄 + USB 接收器直连 X5（`/dev/input/js0`）——优先部署连接便利，无高精度/硬实时要求；SBUS/蓝牙方案废弃
- [ ] IMU（BMI088，达妙板）安装外参在哪端补偿 —— **建议 H723 端一次性转到机体系**
- [x] 两板间通信物理链路： UART

---

## 1. 项目概况

| 项 | 内容 |
|---|---|
| 机器人 | 串联二连杆双足腿（hip_pitch×2 + knee×2，DM-J4340P，MIT 位置控制）+ 
    轮足（DM-H65×2，速度模式，待 §0 冻结），共 6 电机，分left/right两线CANFD通信到STM32H723 |
| 实时层 | 达妙 STM32H723 开发板：1kHz 电机 CANFD 环（MIT 打包、达妙 float↔uint 编码在此侧）、
    BMI088 采集与姿态融合、限幅与看门狗；已有自平衡固件（保留作 STAND/fallback） |
| 上位机 | RDK X5（8×A55，10TOPS BPU/Bayes-e，Ubuntu 22.04 预装 TROS，Python 3.10）；**控制环不使用 ROS2** |
| 策略 | Isaac Lab - rsl_rl，导出 **policy.onnx**（首版限定前馈 MLP） |
| X5 侧目标 | 50Hz 策略环：状态快照 → obs 拼装 → ONNX 推理 → 动作后处理 → 下发；
    含三态环境（dummy/mujoco/serial）、状态机、日志与联调工具 |
| 优先级 | **通信 + 推理 > 视觉**；视觉（深度/高程）为后续独立进程（§14） |

---

## 2. 系统架构与设计原则

```
IsaacLab(PC训练) ─policy.onnx + export_cfg→yaml─►┐
2.4GHz无线手柄 ═USB接收器═ /dev/input/js0 ══════►│
                                                  ▼
                    ┌───────────── 同一套上层代码(状态机/obs/推理/后处理) ─────────────┐
                    │ deploy_dummy.py → DummyEnv   (无硬件无仿真, 测管道)             │
                    │ deploy_sim.py   → MujocoEnv  (sim2sim 安全网, 建议在PC跑)       │
                    │ deploy_real.py  → SerialEnv  (实物)                             │
                    └──────────────────────────────┬───────────────────────────────────┘
                                                   │ USB CDC / UART, 自定义定长帧
                                                   ▼
                                      STM32H723(达妙板, 1kHz)  ◄─CANFD─►  4×DM-J4340P + 2×DM-H65
                                      MIT打包/达妙编码/IMU/限幅/看门狗/自平衡fallback
```

**设计原则（不可违背）**

1. X5 **不接电机 CANFD 总线**；1kHz PD、限幅、超时保护全部在 H723 本地闭环。
2. X5 只发"目标"（统一 MIT 五元组），H723 用最新目标 1kHz 执行；X5 掉线 → H723 看门狗自动降级（阻尼→下电）。
3. **三态环境同接口同行为**：DummyEnv / MujocoEnv / SerialEnv 共用全部上层代码，只换实例化。先 dummy 通管道 → 
        再 MuJoCo 验策略 → 最后上实物。
4. 达妙自平衡固件保留为 STAND/fallback 模式。遥控经 X5 软件链路（手柄→fsm→CmdFrame），**最后一道安全不依赖遥控**：
    手柄或 X5 进程任一失联，均由 H723 看门狗自动降级（阻尼→下电）；整机另配硬件急停/总电开关。
5. 协议**单一真源** `firmware_shared/protocol.h`；sim2real 常量**单一真源** `config/*.yaml`（数值来自训练侧导出）。
6. 单位 SI：rad、rad/s、N·m、m/s²，均为**电机输出轴**量；机体系 FLU；四元数 wxyz。
7. **严禁上电直接进推理**：必须经 状态机过渡（§7）。

---

## 3. 关键技术决策（含理由；改动需记录）

| # | 决策 | 理由 | 备选/备注 |
|---|---|---|---|
| D1 | X5↔H723 链路首选 **USB CDC 虚拟串口** | 零硬件改动；达妙固件已有 VCP；带宽/延迟充裕 | UART TTL（最稳）；CANFD（X5 板载 TCAN4550/SocketCAN，为 SPI 桥接，仅作 X5↔H723 链路，不上电机总线） |
| D2 | 推理引擎 **onnxruntime CPU** 起步 | 小 MLP 单次前向 <1ms；免量化风险 | BPU（OpenExplorer→.bin + hobot_dnn/bpu_infer_lib）留给视觉阶段，届时 locomotion 可一并上 BPU 省 CPU（§14） |
| D3 | 语言 **Python 3.10**，模块边界按可移植设计 | 50Hz 环抖动可接受 | 不达标再移植 C++ |
| D4 | 控制环**无 ROS2**、单进程；感知为独立 TROS 进程 | 确定性与低延迟；感知崩溃不影响行走 | — |
| D5 | 策略频率 **50Hz** = sim dt×decimation（需核对） | 与训练一致 | — |
| D6 | 状态上行 **200Hz** / 指令下行 **50Hz** 起步 | 对 50Hz 策略足够 | 可提至 500Hz（USB 无压力） |
| D7 | **三态环境抽象 BaseEnv**（dummy/mujoco/serial） | sim2sim 安全网 + 无硬件管道自测；上层代码零改动换环境 | 借 RoboJuDo 的 base_env 思想，实现保持精简 |
| D8 | 状态上行用**单一定长 StateFrame（电机+IMU 同帧）** | 原子快照，obs 各项时间一致；定长解析简单 | 舍弃"电机/IMU 分两条消息"方案（时间错位隐患）；扩展消息走 AuxFrame |
| D9 | 上位链路载荷用 **float32**，不用达妙 16/12bit 压缩 | USB 带宽充裕（138B×200Hz≈28KB/s）；避免 P_MAX/V_MAX/T_MAX 双份维护与量化误差 | 达妙编码保留在 H723↔电机侧（固件职责）；若未来改窄带链路再压缩 |
| D10 | 协议统一 **MIT 五元组/电机**，轮子语义由配置切换 | 力矩(kp=kd=0, t_ff=τ) 与 速度(kp=0, v_des+kd) 都能表达；训练改配置即可跟随 | H723 对 kp/kd/t_ff 做**白名单限幅**兜底（防上位机发疯） |
| D11 | **不采用 LeRobot 作为部署框架**；仅采纳其 BPU 工作流与 hbm_runtime | LeRobot 生态定位是机械臂操作的模仿学习/VLA（遥操作数据采集 + ACT/VLA + 总线舵机 + ~30Hz），与腿式 RL 的 50Hz 本体感受紧循环、CANFD 力控、实时安全层不匹配；官方 BPU 工具目前仅验证 ACT@RDK S100 | 评估见附录 E（本项目无机械臂任务规划） |
| D12 | 遥控 = **2.4GHz 无线手柄 + USB 接收器直连 X5**（`/dev/input/js0`，内核 joydev/xpad 免驱） | 即插即用、免配对，部署连接最省事；无高精度/硬实时要求，毫秒级输入延迟对 50Hz 指令绰绰有余 | 蓝牙手柄、SBUS，弃；协议 rc[6] 转预留位 |

---

## 4. 通信协议规范 ICD v0.2（M0 冻结）

### 4.1 链路参数与串口配置

- USB CDC：`/dev/dm_h723`（udev 固定名）；UART 备选 921600-8N1 起步。字节序**小端**，`#pragma pack(1)` 无填充。
- **坑位（必检）**：串口必须纯二进制透传。pyserial 默认 raw 可用；若自行 termios 或经转接芯片，须关 `ICRNL/INLCR/IGNCR/OPOST`，否则 `0x0D↔0x0A` 被篡改、CRC 全崩。

### 4.2 帧格式（唯一真源 `firmware_shared/protocol.h`）

```c
#pragma pack(push, 1)
typedef struct { float p_des, v_des, kp, kd, t_ff; } MotorCmd;   // 20 B, MIT五元组
typedef struct { float q, dq, tau; } MotorFb;                    // 12 B

typedef struct {            // X5 -> H723, 126 B, 50~100 Hz
    uint8_t  head[2];       // {0xAA, 0x55}
    uint8_t  mode;          // 期望模式, §4.5
    uint8_t  seq;
    MotorCmd m[6];          // 顺序 §4.5; 腿=位置目标, 轮=按 §4.6 语义
    uint16_t crc;           // CRC-16/CCITT-FALSE, 覆盖 [0..123]
} CmdFrame;

typedef struct {            // H723 -> X5, 138 B, 200 Hz(可调500) —— 原子快照
    uint8_t  head[2];       // {0xBB, 0x55}
    uint8_t  mode;          // 低4位当前Mode, 高4位FaultCode
    uint8_t  seq;
    uint32_t t_us;          // MCU时间戳(us, 允许回绕)
    float    quat[4];       // w,x,y,z 机体姿态(BMI088融合, 已含安装外参)
    float    gyro[3];       // rad/s 机体系
    float    acc[3];        // m/s^2 机体系
    MotorFb  m[6];          // 输出轴 q/dq/tau
    uint16_t rc[6];         // 预留位(遥控已定为手柄直连X5, 恒为0, 帧布局保持不变)
    float    vbus;
    uint16_t crc;           // 覆盖 [0..135]
} StateFrame;

typedef struct {            // X5 -> H723, 事件式(非周期), 6 B
    uint8_t  head[2];       // {0xCC, 0x55}
    uint8_t  cmd;           // 1=SET_ZERO 2=CLEAR_FAULT (可扩展)
    uint8_t  arg;           // 电机索引等
    uint16_t crc;
} AuxFrame;
#pragma pack(pop)
```

Python `struct` 格式串（单测断言 126/138/6）：

```python
CMD_FMT   = "<2sBB30fH"            # 126 B
STATE_FMT = "<2sBBI4f3f3f18f6HfH"  # 138 B
AUX_FMT   = "<2sBBH"               # 6 B
```

### 4.3 CRC 与帧同步

**CRC-16/CCITT-FALSE**（poly 0x1021, init 0xFFFF, 无反转, xorout 0），测试向量 `crc16(b"123456789")==0x29B1`（写进单测）。接收端环形缓冲扫 head → 定长取帧 → CRC 校验 → 失败前移 1 字节重同步，计数 `crc_err/resync`。

### 4.4 时序与看门狗（安全语义，必须实现）

| 端 | 条件 | 动作 |
|---|---|---|
| H723 | >200ms 无合法 CmdFrame | 切 DAMP（kp=0，kd=1~2） |
| H723 | 再 >1s 仍无 | DISABLE 下电 |
| H723 | kp/kd/t_ff 超白名单范围 | 拒执行该帧并置 FaultCode |
| H723 | \|pitch\|/\|roll\| 超阈值（倾覆） | 本地切 DAMP，上报 F_OVER_TILT |
| X5 | >100ms 无 StateFrame | 停发 RL 动作，状态机进 LINK_LOST |
| X5 | 手柄失联（设备节点消失或读超时 >500ms） | 指令清零并请求 DAMP；H723 看门狗为最终兜底 |

### 4.5 枚举（占位，M0 与固件对齐后冻结）

```c
enum MotorIdx { M_L_HIP=0, M_L_KNEE=1, M_R_HIP=2, M_R_KNEE=3, M_L_WHEEL=4, M_R_WHEEL=5 }; // TODO对齐CAN ID
enum Mode     { MODE_DISABLE=0, MODE_DAMP=1, MODE_SQUAT=2, MODE_STAND=3 /*板载自平衡或固定姿态*/, MODE_RL=4 };
enum Fault    { F_OK=0, F_CMD_TIMEOUT=1, F_MOTOR_LOST=2, F_OVER_TEMP=3, F_OVER_TILT=4, F_IMU=5, F_BAD_GAIN=6 };
```

### 4.6 轮足两种语义 → 同一五元组（由 `config/robot.yaml: wheel_mode` 切换）

| wheel_mode | p_des | v_des | kp | kd | t_ff |
|---|---|---|---|---|---|
| `torque`（力矩/电流） | 0 | 0 | 0 | 0 | τ = action×scale（限幅） |
| `velocity`（速度） | 0 | action×scale | 0 | kd_wheel | 0 |

腿恒为：p_des = default + action×scale（软限幅），kp/kd 取自 robot.yaml（=训练 stiffness/damping）。

---

## 5. 仓库结构与模块规格

```
rdkx5_deploy/
├── CLAUDE.md
├── docs/PLAN.md                     # 本文档
├── firmware_shared/protocol.h       # 帧定义唯一真源
├── config/
│   ├── robot.yaml                   # 关节映射/符号/零位/kp kd/限幅/wheel_mode/squat与stand姿态
│   ├── policy.yaml                  # 由 tools/export_cfg.py 从训练env生成(obs顺序/scales/action/clip/dt)
│   └── comm.yaml                    # 端口/频率/超时
├── assets/
│   ├── policy.onnx
│   └── robot.xml                    # MJCF(sim2sim)
├── src/
│   ├── comm/
│   │   ├── protocol.py              # pack/unpack + CRC16(纯函数, 100%单测)
│   │   └── serial_link.py           # 读线程/环形缓冲/帧同步/统计/自动重连
│   ├── envs/                        # ★三态环境(接口一致, 上层无感)
│   │   ├── base_env.py
│   │   ├── dummy_env.py             # 假状态(站姿+重力[0,0,-1]+可选噪声), 测管道
│   │   ├── mujoco_env.py            # 加载MJCF, 内部1kHz PD子步执行五元组, 50Hz外环
│   │   └── serial_env.py            # 组合 SerialLink+protocol, 实物
│   ├── policy/
│   │   ├── obs_builder.py           # 严格按policy.yaml拼obs(映射/缩放/可选历史)
│   │   ├── onnx_policy.py           # ort session(CPU, intra_op=1), warmup, 耗时统计
│   │   └── action_mapper.py         # action→(6,5)五元组; 腿/轮两支路; clip与限幅
│   ├── control/
│   │   ├── fsm.py                   # IDLE/DAMP/SQUAT/STAND/RL_RUN/LINK_LOST/FAULT
│   │   └── loop.py                  # 50Hz主环(monotonic定时, 抖动统计)
│   ├── io/
│   │   ├── gamepad.py               # 2.4GHz手柄: /dev/input/js0(joydev)读取 → cmd_vel+键位; 死区/限幅/失联检测/热插拔重连
│   │   └── logger.py                # 每周期(t,obs,action,state,jitter)落盘→npz
│   ├── safety.py                    # X5侧检查(超时/倾角/指令限幅)→模式降级
│   ├── deploy_dummy.py              # 入口: DummyEnv
│   ├── deploy_sim.py                # 入口: MujocoEnv(建议在PC跑)
│   └── deploy_real.py               # 入口: SerialEnv
├── tools/
│   ├── export_cfg.py                # (训练机跑)IsaacLab env cfg → policy.yaml, 消灭手抄scale
│   ├── link_bench.py  state_monitor.py  joint_check.py  stand_hold.py
│   ├── replay_check.py              # sim导出obs.npy→板上推理→与action.npy逐位比对
│   └── plot_log.py
└── tests/                           # pytest: CRC向量/pack↔unpack/布局尺寸/obs数值对齐/三env一致性冒烟
```

**BaseEnv 抽象接口（三个实现行为必须一致；`step()` 直接吃五元组，与协议同构）**

```python
class BaseEnv(ABC):
    def self_check(self) -> bool: ...          # 上电自检(链路/电机/IMU就绪; dummy恒True)
    def reset(self) -> None: ...
    def update(self) -> None: ...              # 刷新最新状态快照(读帧/步进仿真/造假)
    def step(self, targets: np.ndarray) -> None: ...   # (6,5) MIT五元组
    def set_zero(self, motor: int) -> None: ...        # 实物→AuxFrame; 仿真no-op
    def shutdown(self) -> None: ...            # 安全下电(阻尼→失能)
    # update()后可读属性:
    # dof_pos(6), dof_vel(6), base_quat(wxyz), base_ang_vel(3), base_lin_acc(3),
    # link_ok: bool, fault: int, rc(6)
```

**其余接口签名与控制环硬性约束沿用 v0.1**：`crc16_ccitt / pack_cmd / try_parse_state / SerialLink(latest_state,send,stats) / ObsBuilder.build / ActionMapper.to_motor_targets`；环内禁止阻塞 IO、print、大对象分配；魔法数一律进 yaml。

---

## 6. 配置文件规范

骨架同 v0.1（robot/policy/comm 三分），新增/强调：

```yaml
# robot.yaml 增补
wheel_mode: torque            # torque | velocity —— 必须 = 训练env的actuator配置
postures:                     # 状态机姿态目标(腿4维, rad)
  squat: {L_hip: 0.0, L_knee: 0.0, R_hip: 0.0, R_knee: 0.0}   # TODO
  stand: {...}
action_filter: {enabled: false, alpha: 0.9}   # 仅当训练侧存在同样滤波才开

# policy.yaml —— 由 tools/export_cfg.py 从 IsaacLab env cfg 自动生成
```

**scale 占位规则（重要）**：obs_scales / action_scale 的**缩放逻辑在代码里一律保留**；`dummy_env` 阶段数值可用 1.0 占位（只测管道）。**但 1.0 不是可用值**——scale 是训练时固化的契约，**一旦加载真实策略（sim2sim 或实物），必须换成训练真值**，两端不一致一上电就发散。消灭手抄错误的正解是 `export_cfg.py` 从训练配置直接生成 policy.yaml（单一数据源；也可选择把 scale 烘焙进 onnx 计算图，二选一，数值必须来自训练）。

---

## 7. 控制主环与状态机

### 7.1 状态机（含 SQUAT 过渡；**严禁上电直接进 RL**）

```
IDLE ──X──► DAMP ──X──► SQUAT ──X──► STAND(板载自平衡/固定姿态) ──Y──► RL_RUN
  ▲                                        ▲───────────Y──────────────────┘
  A键(任意态): 一键切DAMP;  链路超时→LINK_LOST(停发+阻尼请求);  故障→FAULT
```

键位（2.4GHz 手柄，Xbox 布局）：X 切姿态、Y 起停推理、A 一键阻尼、B 指令清零；左摇杆→vx、右摇杆→ωz（死区 0.1，线性映射后按 policy.yaml 限幅）。手柄失联→指令清零+请求 DAMP；最终兜底为 H723 看门狗与硬件急停（不依赖软件链路）。

### 7.2 50Hz 主环（三态环境共用）

```python
env = make_env(args.env)            # dummy | mujoco | serial
assert env.self_check()
t_next = time.monotonic()
while running:
    env.update()
    fsm.step(events, env, link_stats)
    if fsm.mode == RL_RUN:
        obs = obs_builder.build(env, cmd_vel, last_action)
        a   = np.clip(policy(obs), -clip, clip)
        env.step(mapper.to_motor_targets(a)); last_action = a
    else:
        env.step(mapper.posture_targets(fsm.mode))
    logger.log(...)
    t_next += 0.02; time.sleep(max(0, t_next - time.monotonic()))
```

### 7.3 观测组装（示例 24 维；**一切以你的 env cfg 为准**）

| 项 | 维度 | 来源/说明 |
|---|---|---|
| base_ang_vel | 3 | ωx,ωy,ωz  gyro（机体系） |
| projected_gravity | 3 | x,y,z 重力向量 |
| commands | 2 | vx, ωz（此构型通常无 vy；维度照抄 env） |
| joint_pos_rel | 4 | 仅 4 个腿关节（q−default）；**轮角无界，绝不进 obs** |
| joint_vel | 6 | 腿 4 + 轮 2 |
| last_action | 6 | 策略上一帧输出 |

若训练用了观测历史/高度扫描，部署侧复刻（history buffer / height_scan 通道，§14）。

---

## 8. sim2real 对齐清单（逐项打勾）

1. [ ] **关节顺序**：Isaac Lab（USD 解析序）↔ MJCF ↔ 电机 ID 三方经 robot.yaml 映射统一——已知高发坑，sim2sim 阶段即可暴露。
2. [ ] 符号/零位经 `joint_check` 实测确认；零位与 URDF 一致。
3. [ ] kp/kd = 训练 ImplicitActuator stiffness/damping（输出轴、rad）；X5 20ms 发目标，H723 1kHz PD。
4. [ ] **轮足语义**（torque/velocity）与训练一致（§4.6）。
5. [ ] obs_scales/action_scale/clip/default_pos 来自 `export_cfg.py` 生成的 policy.yaml，禁手抄；滤波/历史与训练一致。
6. [ ] 部署频率 = sim dt×decimation（=0.02s 核对）。
7. [ ] IMU：安装旋转 H723 端补偿；gyro 直接进 obs；重力方向用四元数不用欧拉角；四元数 wxyz 约定两端一致。
8. [ ] 串口二进制透传核查（§4.1 坑位）。
9. [ ] 验证链：dummy 管道 → replay_check 数值一致性 → **MuJoCo sim2sim** → 挂架手扳关节 → 挂架 RL → 落地；全程日志落盘。

---

## 9. 里程碑与验收标准

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M0** 协议+管道 | ICD v0.2 冻结；protocol/CRC/pack↔unpack + 单测；**dummy_env 跑通全管道**（状态机+obs+推理调用，scale 可 1.0） | pytest 全绿（CRC=0x29B1、尺寸 126/138/6）；deploy_dummy 连续跑 10min 无异常，环抖动 p99<3ms |
| **M1** 链路压测 | link_bench：上行 200Hz/下行 50Hz×30min | crc_err=0、resync=0、RTT p99<5ms；拔插自动恢复 |
| **M2** 上行语义 | state_monitor + joint_check | 全关节方向/单位/顺序确认；静置 gravity≈[0,0,−1]±0.05 |
| **M3** 下行控制 | 单电机点动 → 轮子按 wheel_mode 驱动 → stand_hold | 站姿保持 1min 无震荡；看门狗演练（断链→阻尼）通过 |
| **M4** sim2sim | export_cfg 生成真值 yaml；replay_check；deploy_sim 在 MuJoCo 走起来 | 板上 vs sim 动作 max\|diff\|<1e-4；MuJoCo 内速度跟踪不倒（此步起 scale 必为训练真值） |
| **M5** 实物空跑 | 挂架：SQUAT→STAND→RL 零指令→小指令 | 空跑 5min 不发散 |
| **M6** 落地 | 低速行走/平衡 | 三安全链演练通过：H723 看门狗、遥控急停、倾覆检测 |
| **M7** 越野迭代 | 场景渐进 + 日志复盘；（可选）BPU/视觉预研 | — |

> M4 可与 M1–M3 并行（sim2sim 在 PC 上跑，不占硬件）。

---

## 10. 联调工具规格（tools/）

沿用 v0.1 全套（link_bench / state_monitor / joint_check / stand_hold / replay_check / plot_log，均可独立运行、输出回贴给 Claude Code 分析），新增：

- `export_cfg.py`（**训练机上跑**）：读 IsaacLab env cfg → 生成 `config/policy.yaml`（obs 顺序/scales/action/clip/dt/关节名序），并顺带导出一段 sim 轨迹 `obs.npy/act.npy` 供 replay_check 用。

---

## 11. 运行环境与部署

**系统基线（按官方《快速开始》v3.5.0）**：RDK OS ≥ 3.5.0（Ubuntu 22.04 aarch64，预装 TROS-Humble，与 rdk_model_zoo rdk_x5 分支要求一致），烧录与首次启动按官方 1.2 系统烧录 / 1.3 入门配置；`srpi-config` 统一管理 Wi-Fi、SSH/VNC、40pin 外设总线、CPU 超频与 ION 内存——**若启用 UART 备选链路，须先在 srpi-config（或 /boot/config.txt dtoverlay）使能对应 40pin 串口**；ION 留默认，未来上 BPU 视觉再调。日常部署走 SSH/SCP（或 VSCode Remote-SSH / RDK Studio）；建议保留一根串口调试线，作为野外脱网时的救援通道（官方 1.4 远程登录：串口/VNC/SSH）。

```bash
# RDK X5 控制环最小集
pip install onnxruntime numpy pyserial pytest evdev
# PC(sim2sim): pip install mujoco onnxruntime numpy
# 未来视觉才需要: OpenExplorer工具链 + bpu_infer_lib_x5 / hobot_dnn / hbm_runtime; TROS已预装(/opt/tros/humble)
```

udev 固定设备名、taskset 绑核、systemd 单元、策略导出流程（rsl_rl `export_policy_as_onnx`，PC 上先比对 torch vs ort 输出）——沿用 v0.1 §11；CPU 解锁满频并锁 performance（官方推荐姿势）：`echo 1 > /sys/devices/system/cpu/cpufreq/boost`（8×A55@1.8GHz）+ `echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor`。BPU 导出规则（届时用）：opset=11、静态 shape、batch=1、simplify；`hb_mapper checker/makertbin --march bayes-e`；板端推理库优先 `hbm_runtime`（`pip install hbm-runtime`，官方 rdk_LeRobot_tools 即用它），bpu_infer_lib / hobot_dnn 备选；整套「导出脚本一键产出 ONNX+校准数据+编译脚本、产物打包归一化 .npy 与精度验证 .npy」的工作流仿照 rdk_LeRobot_tools 的 `export_bpu_actpolicy.py`（详见附录 E）。LSTM 策略暂不支持（需显式 h/c，列为后续任务）。

---

## 12. 与 Claude Code 的协作约定

1. **开发顺序**：protocol(+tests) → base/dummy_env → fsm/obs/mapper(+tests) → deploy_dummy 跑通（M0）→ serial_link(+link_bench) → serial_env → tools 各脚本 → mujoco_env(deploy_sim) → 实物里程碑。每个 PR 对应一个模块。
2. **单一真源**：帧布局只认 `protocol.h`；sim2real 常量只认 yaml（且 policy.yaml 由 export_cfg 生成，禁手改数值）。
3. **可测优先**：CRC、pack/unpack、obs_builder、action_mapper、fsm 必须有 pytest；三态环境加"同输入同输出"冒烟测试。
4. **硬件在环边界**：M1–M6 实机步骤由人跑 tools/ 脚本，日志/npz 回贴给 Claude Code 分析；Claude Code 不假设硬件行为。
5. 控制环代码规范：无阻塞 IO、无 print、预分配数组；异常必须导致安全降级而非崩溃退出。
6. 实机异常处理流程：plot_log 复盘 → 归因（协议/映射/scale/增益/策略）→ 改动记 docs/CHANGELOG。

---

## 13. 风险与备选

| 风险 | 触发信号 | 备选 |
|---|---|---|
| USB 振动/EMI 不稳 | link_bench 出现 resync/掉线 | 切 UART；再不行切 CANFD（SocketCAN），协议层不变 |
| Python 环抖动大 | jitter p99>3ms | 主环+推理移植 C++，协议/配置复用 |
| 轮足语义与训练不符 | sim2sim 就发散 | 改 wheel_mode 或回训练侧统一 |
| 关节顺序/符号错 | sim2sim 倒地 / joint_check 不一致 | 只改 robot.yaml 映射，禁多处硬编码 |
| scale 手抄错 | 一上电发散 | export_cfg 单一数据源（制度性消灭） |
| 策略含 LSTM | 导出报隐状态 | 显式 h/c 维护或回训练改 MLP |
| 2.4GHz 接收器受电机/电调干扰或松脱 | 手柄读超时/键值冻结 | 指令清零+DAMP；用 USB 延长线把接收器远离功率器件；H723 看门狗兜底 |

---

## 14. 感知层（后续，独立进程）

TROS 起独立进程（`mipi_cam` + 双目深度/高程节点，BPU 推理走 hobot_dnn），结果经 **UDP/共享内存** 喂给控制环的 `height_scan` 通道；控制环保持非 ROS2 的确定性，感知崩溃不影响站立行走。越野优先深度相机→elevation mapping（2D 激光对地形帮助有限）；训练侧同步开崎岖地形课程 + 高程扫描。届时 locomotion 策略可与视觉网络一并量化上 BPU。

**参考仓库定位**：RoboJuDo（主架构范本：base_env 三态 + policy 抽象 + 配置驱动；真机层是 Unitree DDS，serial_env 需自研）、unitree_rl_gym（轻量 deploy_mujoco/deploy_real 起点）、robot_land（RDK 平台状态机/服务化；其 bridge 为纯位置协议，轮足部分不可照搬）、rdk_model_zoo（rdk_x5 分支为 X5 主交付分支：视觉模型首查处，hbm_runtime/C++ 统一范例；S 系列分支 rdk_model_zoo_s 的 Planning 样例含 walk-these-ways-go2 / ASAP / LeRobot_ACTpolicy，是腿式/策略网络上 BPU 的官方先例）、rdk_LeRobot_tools + D-Robotics/lerobot（仅取其策略→BPU 的官方工作流范本，评估见附录 E）、nodehub / TROS（视觉组件）。

---

## 附录 A：CRC-16/CCITT-FALSE 参考实现（同 v0.1，测试向量 0x29B1）

```python
def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc
```

## 附录 B：projected_gravity

机体系重力方向 = R(q)ᵀ·[0,0,−1] = `[2(wy−xz), −2(wx+yz), 2(x²+y²)−1]`（q=[w,x,y,z] 单位四元数）。自检：q=[1,0,0,0]→[0,0,−1]；绕 +y 前俯 θ→[sinθ,0,−cosθ]。

## 附录 C：约定速查

机体系 FLU；rad / rad/s / N·m（输出轴）；四元数 wxyz；小端；策略周期 0.02s；轮角不进 obs。

## 附录 D：两版路线合并取舍记录

| 另一版的做法 | 取舍 | 理由 |
|---|---|---|
| BaseEnv 三态环境（dummy/mujoco/serial）+ sim2sim 安全网 | **采纳** | 无硬件即可验证管道与策略；关节顺序/scale 类错误提前到 MuJoCo 暴露 |
| dummy_env 先行、deploy_{dummy,sim,real} 三入口共用上层 | **采纳** | 并入 M0 验收 |
| scale 先填 1.0 占位 + "训练真值契约"强警告 | **采纳并加强** | 增加 `export_cfg.py` 从训练配置直接生成 yaml，制度性消灭手抄 |
| 轮足走力矩/电流、下行拆腿/轮两条消息 | **半采纳** | 轮足语义改为**配置项且以训练为准**（torque/velocity 五元组均可表达）；不拆消息，保持单帧 |
| MAVLink v2 精简帧 + 电机/IMU 分消息（0x03/0x04） | **不采纳** | 分消息导致 obs 时间错位；定长双帧解析更简、快照原子；扩展需求由 AuxFrame 承担 |
| 上位链路用达妙 float↔uint（16/12bit）压缩 | **不采纳** | USB 带宽充裕；避免 P_MAX/V_MAX/T_MAX 双份维护与量化误差；该编码属 H723↔电机侧固件职责 |
| Kp/Kd 固定在 MCU 侧 | **半采纳** | 协议保留五元组（灵活、显式），H723 端加 kp/kd/t_ff **白名单限幅**兜底（F_BAD_GAIN） |
| 动作低通滤波（0.9/0.1） | **条件采纳** | 作为配置项默认关闭；仅当训练侧存在同样滤波才开 |
| SQUAT 状态 + X/Y/A 键位 + 严禁直接进推理 | **采纳** | 并入 §7 状态机 |
| set_zero（标零位）指令 | **采纳** | 以 AuxFrame 实现 |
| termios 二进制透传坑（ICRNL/OPOST） | **采纳** | 写入 §4.1 与对齐清单 |
| TROS 感知独立进程 + UDP/shm → height_scan | **采纳** | 并入 §14 |
| BPU 导出规则（opset11/静态shape/simplify）与量化时机 | **采纳** | 并入 §11 |
| 缺失项：看门狗语义/故障码/验收标准/CRC 向量/日志回放/udev/systemd | **保留本方案** | 安全与可复盘必需，另一版未覆盖 |

## 附录 E：官方 LeRobot 生态评估（v0.3 审视，不重构）

### E.1 三个仓库是什么

| 仓库 | 定位 | 与本项目的关系 |
|---|---|---|
| D-Robotics/lerobot | HF LeRobot 官方 fork：「在 RDK 系列上运行 LeRobot 并用 BPU 加速推理」。LeRobot 本体面向真实机器人**操作任务的模仿学习/VLA**：遥操作数据采集、LeRobotDataset、ACT/Diffusion/VLA 策略库、feetech/dynamixel 舵机 + USB 相机设备层；fork 锁定 datasets 依赖版本保证板端兼容 | 任务域不重叠（E.2），不引入 |
| rdk_LeRobot_tools | 「BPU Tools for LeRobot」：把 LeRobot 训练的 **ACT** 策略导出并编译到 BPU。链路：开发机 `export_bpu_actpolicy.py`（PyTorch→ONNX + 校准数据 + 自动生成 `build_all.sh`，按 `bayes-e`(X5) / `nash`(S100) 平台字段自动配编译参数）→ OE Docker 编译 → `bpu_output`（.hbm/.bin + 归一化 .npy + `new_actions.npy` 精度验证）→ 板端 `bpu_control_robot.py` + `hbm_runtime`（默认 so101 机械臂，30Hz）。**官方注明目前仅在 RDK S100 上验证过 ACT，其他平台/模型不保证** | 部署框架不采用；**工作流范式采纳**（E.3） |
| rdk_model_zoo | 官方 BPU 模型仓库（rdk_x5 分支为 X5 主交付分支）：预量化模型 + Python(`hbm_runtime`)/C++ 统一接口范例，覆盖分类/检测/分割/姿态/OCR/多模态；PTQ：ONNX→bin，QAT：pt→hbm。S 系列分支 rdk_model_zoo_s 另有 Planning 类样例：LeRobot_ACTpolicy、walk-these-ways-go2、ASAP（策略网络上 BPU 的官方先例） | 未来视觉模型首查处；策略上 BPU 时参考其 Planning 样例 |

### E.2 为什么不用 LeRobot 重构本方案

1. **任务域不同**：LeRobot 解决「真人示教→数据集→IL/VLA 训练→机械臂执行」；本项目是「IsaacLab RL→本体感受策略→腿轮运动控制」，没有数据采集、数据集、IL 训练环节，LeRobot 的核心资产（LeRobotDataset、teleop、策略库）全部用不上。
2. **控制形态不同**：LeRobot 设备层是总线舵机 + USB 相机、~30Hz 位置伺服；本项目是 CANFD 力控电机 + 自研 STM32 实时层、50Hz MIT 目标 + 1kHz PD、看门狗/状态机/倾覆保护——这些 LeRobot 均不提供，套框架只增加适配成本，还要为它写自定义 Robot/MotorsBus，工作量不减反增。
3. **工具边界**：rdk_LeRobot_tools 当前仅验证 ACT@S100，且面向含视觉编码器的策略；本项目的 MLP RL 策略在 X5 CPU 上 onnxruntime <1ms，BPU 化无收益（D2 结论不变）。
4. **官方实践反而佐证本方案**：`bpu_control_robot.py` 本身就是无 ROS2 的板端 Python 控制循环，与 deploy_real 同构——「控制环单进程、无 ROS2」与官方做法一致。

### E.3 吸收清单（已并入正文）

| 吸收项 | 落点 |
|---|---|
| `hbm_runtime` 为板端 BPU 推理库首选（`pip install hbm-runtime`） | §11 |
| 「导出脚本一键产出 ONNX+校准数据+编译脚本；产物打包归一化 .npy + 精度验证 .npy」的打包范式（与本方案 replay_check 的一致性验证思路互相印证） | §11；未来仿写 `export_bpu_locopolicy.py` |
| `bpu_export_config.yaml` 的平台字段（bayes-e=X5）自动配编译参数 | §11 |
| rdk_model_zoo(_s) Planning 样例 = 策略网络上 BPU 的官方先例 | §14 |

### E.4 触发重评的条件

官方在 X5 上验证通过更通用的策略 BPU 部署链路（届时按 E.3 范式迁移 locomotion 策略）。
