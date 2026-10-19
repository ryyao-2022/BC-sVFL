# 加密接口

加密后端负责计算客户端表示与服务器首层权重的内积。训练过程通过统一接口调用后端，模型结构与优化器无需随加密方法变化。

## 1. 接口与数据形状

```python
setup(dimension: int, client: str) -> (public, authority)
encrypt(public, z: np.ndarray, context: Context) -> ciphertext
keygen(authority, weights: np.ndarray, context: Context) -> keys
evaluate(public, ciphertext, keys) -> np.ndarray
```

- `z`：`[batch, dimension]`，客户端表示。
- `weights`：`[server_first_width, dimension]`，服务器首层的客户端权重块。
- 输出：`[batch, server_first_width]`，对应 `z @ weights.T`。

接口使用 CPU NumPy 数组，输出转回模型所在设备。`public` 为公开参数，`authority` 为密钥机构的主密钥或服务句柄。客户端调用 `encrypt`，密钥机构调用 `keygen`，服务器使用公开参数、密文和功能密钥调用 `evaluate`。

`Context` 包含客户端、前向批次和更新前模型版本；参考后端同时检查初始化标识。每次重算表示使用新加密随机数，功能密钥与当前权重对应。`ForwardTransport` 在单进程中调度这些角色并记录耗时。

## 2. 内积 FE

参考后端实现四个步骤：

1. **Setup：**在素数阶子群中选 `g,h`，采样 `s_i,t_i`，发布 `h_i=g^s_i h^t_i`。
2. **KeyGen：**按权重行派生 `s_w=Σs_i w_i mod q`、`t_w=Σt_i w_i mod q`。
3. **Enc：**采样 `r`，生成 `C=g^r`、`D=h^r`、`E_i=g^z_i h_i^r`。
4. **Dec：**计算 `ΠE_i^w_i / (C^s_w D^t_w)`，恢复指定范围内的有符号内积。

`rfc3526_2048` 使用 [RFC 3526 第 3 节](https://www.rfc-editor.org/rfc/rfc3526.html#section-3) 的模数，子群阶 `q=(p-1)/2`，`g=4`。`toy31` 使用小整数群，仅用于测试。Python 后端用于研究实验，其大整数运算不是常数时间实现。

## 3. 定点编码

```text
z_int = rint(z * z_scale)
w_int = rint(w * w_scale)
result = integer_inner_product / (z_scale * w_scale)
```

`rint` 取最接近整数，恰好一半时取偶数。负数按模 `q` 表示，恢复时采用有符号区间。

输入应满足 `abs(z)≤z_bound`、`abs(w)≤w_bound`，内积恢复界为：

```text
B = dimension * ceil(z_bound * z_scale) * ceil(w_bound * w_scale)
```

要求 `2B<q`。恢复采用 baby-step/giant-step，表大小约为 `sqrt(2B)`，受 `max_baby_steps` 控制。超出范围、非有限值或维度不匹配时返回错误；调整精度或取值范围也会改变恢复成本。

长度为 d 的内积，其定点误差可按以下上界估计：

```text
error <= d * (w_bound/(2*z_scale) + z_bound/(2*w_scale)
              + 1/(4*z_scale*w_scale))
```

## 4. 与训练流程连接

各客户端的线性贡献相加后，由服务器添加一次偏置并继续计算。反向使用保留的浮点状态：

```text
G_k = Delta.T @ Z_k    # 客户端计算首层权重梯度
H_k = Delta @ W_k      # 服务器计算表示梯度
```

全部梯度均使用更新前参数。FE 定点前向采用忽略舍入导数的直通反向近似；明文后端可用于对照普通 autograd。FE 保护前向首层计算，线性输出和应用层梯度仍按训练协议交换。

## 5. 添加或替换后端

实现一个可导入类，构造函数接收 `crypto.options`，并实现四个接口。以下配置放在 `configs/` 中即可运行：

```json
{
  "extends": "fe_smoke.json",
  "crypto": {
    "backend": "examples.backend_plugin:Backend",
    "options": {"group": "toy31", "z_scale": 100, "w_scale": 100}
  }
}
```

示例适配器调用 Python FE。接入 Go、其他语言库或远程服务时，在适配器内部使用 FFI、RPC 或子进程调用，并保持以下约定：

- 密钥机构保存主密钥或私有句柄，服务器求值使用公开参数、密文与功能密钥。
- 数组布局、缩放、负号约定和错误返回保持一致；通过 JSON 传递大整数时使用字符串。
- 后端输出须满足约定的线性求值语义；改变协议时同步定义数值与梯度接口。
- 接入后运行整数内积、上下文、梯度和训练测试，并记录精度、输入形状与模块耗时。
