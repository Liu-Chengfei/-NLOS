"""轻量校验器。

职责：
    集中放最常见的非空、键集合、数值范围、形状和可迭代检查，供
    协议层、配置层和输入整理层复用。

上游依赖：
    - Python 标准库 collections.abc / numbers — 容器类型与数值类型判断

下游调用者：
    - liquidloc.common.__init__      — 统一导出校验函数给上层
    - liquidloc.protocol.*           — 协议层用校验函数做字段与范围约束
    - liquidloc.pipelines.*          — 流水线层用校验函数做输入验证
    - liquidloc.common.prepared_inputs — 用校验函数验证加载的数据结构
    - liquidloc.estimators.*         — 估计器层用校验函数做数值与形状校验
    - liquidloc.sensors.*            — 传感器层用校验函数做数据校验与转换
    - liquidloc.scenarios.*          — 场景层用校验函数做参数校验
    - liquidloc.models.*             — 模型层用校验函数做特征值转换
    - liquidloc.analysis.*           — 分析层用校验函数做输入验证

核心变量：
    - 无模块级变量，全部通过函数实现
"""

from __future__ import annotations  # 允许类型注解使用现代写法。

import math  # 用于 isfinite 检查。
from collections.abc import Iterable, Mapping, Sequence  # 用来判断容器类型。
from numbers import Real  # 用来判断实数类型（排除 complex）。
from typing import Any  # 用于类型注解。

__all__ = (  # 明确模块对外允许导出的名字。
    "coerce_finite_scalar",
    "is_bool_like",
    "is_integer",
    "is_numeric",
    "is_real",
    "is_string_like",
    "normalize_optional_string",
    "quality_below_floor",
    "require_in_range",
    "require_iterable",
    "require_keys",
    "require_not_none",
    "require_shape",
    "dedupe_preserve_order",
    "validate_path_component",
)


def is_bool_like(value: Any) -> bool:  # 判断值是否为布尔类型（含 numpy.bool_）。
    """判断值是否为 Python bool 或 numpy.bool_。

    numpy.bool_ 在 NumPy 2.x 中不再是 bool 的子类，必须单独检查。
    此函数是项目中所有"排除布尔值"逻辑的统一入口，避免每个文件
    独立实现 try/import numpy 模式。

    Args:
        value: 要检查的值。

    Returns:
        True 如果值是 bool 或 numpy.bool_，否则 False。
    """
    if isinstance(value, bool):  # Python bool 必须先检查（bool 是 int 子类）。
        return True
    try:  # numpy.bool_ 延迟检查，避免硬依赖。
        import numpy as np
        if isinstance(value, np.bool_):
            return True
    except ImportError:
        pass
    return False


def is_numeric(value: Any) -> bool:  # 判断值是否为数值类型（排除 bool 和 numpy.bool_）。
    """判断值是否为数值类型（int/float 及 numpy 对应标量），同时排除 bool 和 numpy.bool_。

    bool 是 int 的子类，numpy.bool_ 在某些版本也继承 int，
    因此必须在 isinstance 检查之前显式排除。
    覆盖 Python int/float 和 numpy 整数/浮点标量（np.int64、np.float64 等）。

    Args:
        value: 要检查的值。

    Returns:
        True 如果值是数值但不是布尔，否则 False。
    """
    if is_bool_like(value):  # 先排除所有布尔类型。
        return False
    if isinstance(value, (int, float)):  # Python 原生数值。
        return True
    try:  # numpy 数值标量延迟检查，避免硬依赖。
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):  # numpy 整数和浮点标量。
            return True
    except ImportError:
        pass
    return False


def is_integer(value: Any) -> bool:  # 判断值是否为整数类型（排除 bool 和 numpy.bool_）。
    """判断值是否为整数类型，同时排除 bool 和 numpy.bool_。

    覆盖 Python int 和 numpy 整数标量（np.int64 等）。
    numpy 2.0+ 中 np.int64 不再是 int 的子类，必须单独检查。

    Args:
        value: 要检查的值。

    Returns:
        True 如果值是整数但不是布尔，否则 False。
    """
    if is_bool_like(value):  # 先排除所有布尔类型。
        return False
    if isinstance(value, int):  # Python 原生整数。
        return True
    try:  # numpy 整数标量延迟检查，避免硬依赖。
        import numpy as np
        if isinstance(value, np.integer):
            return True
    except ImportError:
        pass
    return False


def is_real(value: Any) -> bool:  # 判断值是否为实数类型（排除 bool 和 numpy.bool_）。
    """判断值是否为 numbers.Real 类型，同时排除 bool 和 numpy.bool_。

    使用 numbers.Real ABC 做类型判断，自动排除 complex 和 numpy.complexfloating。
    比 is_numeric 更宽泛，覆盖所有注册为 numbers.Real 的类型（含第三方数值库）。
    注意：decimal.Decimal 未注册 numbers.Real，因此 is_real(Decimal('1.0')) 返回 False。

    Args:
        value: 要检查的值。

    Returns:
        True 如果值是实数但不是布尔，否则 False。
    """
    if is_bool_like(value):  # 先排除所有布尔类型。
        return False
    return isinstance(value, Real)  # numbers.Real 自动排除 complex 和 numpy.complexfloating。


def is_string_like(value: Any) -> bool:  # 判断值是否为字符串类型（含 numpy.str_）。
    """判断值是否为 Python str 或 numpy.str_。

    numpy.str_ 在 NumPy 2.x 中不再是 str 的子类，必须单独检查。
    此函数是项目中所有"字符串类型检查"逻辑的统一入口，避免每个文件
    独立实现 try/import numpy 模式。

    Args:
        value: 要检查的值。

    Returns:
        True 如果值是 str 或 numpy.str_，否则 False。
    """
    if isinstance(value, str):  # Python str 先检查。
        return True
    try:  # numpy.str_ 延迟检查，避免硬依赖。
        import numpy as np
        if isinstance(value, np.str_):
            return True
    except ImportError:
        pass
    return False


def normalize_optional_string(value: Any) -> str | None:
    """把可选字符串规整成去空白后的 `str`，不合法时返回 `None`。

    Args:
        value: 可能是字符串、空值或其它类型的输入。

    Returns:
        去空白后的字符串；如果输入不是字符串或去空白后为空，则返回 `None`。
    """
    if not is_string_like(value):
        return None
    normalized_value = str(value).strip()
    return normalized_value or None


def validate_path_component(value: str, *, name: str) -> str:
    """校验路径组件不包含穿越字符、空字符串和 null 字节。

    Args:
        value: 待校验的路径组件字符串。
        name: 参数名，用于错误消息。

    Returns:
        校验通过的原始字符串。

    Raises:
        TypeError: value 不是 str 类型。
        ValueError: value 为空字符串、包含 null 字节或路径穿越字符。
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if '\x00' in value:
        raise ValueError(f"{name} must not contain null bytes: {value!r}")
    if value == '.':
        raise ValueError(f"{name} must not be a current directory reference: {value!r}")
    if '..' in value or '/' in value or '\\' in value:
        raise ValueError(f"{name} must not contain path traversal characters: {value!r}")

    return value


def require_not_none(value: Any, name: str) -> None:  # 校验值不能是 None。
    """确保值不是 `None`。

    Args:
        value: 要检查的值，可以是任意类型。
        name (str): 值的名称，用于错误消息中标识哪个参数为空。

    Raises:
        ValueError: 当 value 为 None 时抛出。
    """
    if value is None:  # 如果值为空。
        raise ValueError(f"{name} must not be None")  # 直接报错。


def require_keys(mapping: Any, required_keys: Sequence[str], *, name: str = "mapping") -> None:  # 校验映射包含指定键。
    """确保映射里包含指定的所有键。

    Args:
        mapping: 要检查的映射对象，必须是 collections.abc.Mapping 的实例。
        required_keys (Sequence[str]): 必须存在的键名列表。
        name (str): 映射的名称，用于错误消息，默认为 "mapping"。

    Raises:
        ValueError: 当 mapping 为 None 时抛出。
        TypeError: 当 mapping 不是 Mapping 类型时抛出。
        KeyError: 当 mapping 缺少 required_keys 中的某些键时抛出。

    Note:
        当 required_keys 为空序列时，函数静默通过（无要求则无违规）。
    """
    require_not_none(mapping, name)  # 先排除 None。
    if not isinstance(mapping, Mapping):  # 必须是映射类型。
        raise TypeError(f"{name} must be a mapping, got {type(mapping).__name__}")  # 类型不对就报错。
    missing = [key for key in required_keys if key not in mapping]  # 找出缺失的键。
    if missing:  # 只要缺键就不能继续。
        raise KeyError(f"{name} is missing required keys: {missing}")  # 明确指出缺失项。


def require_in_range(
    value: Any,
    name: str,
    *,
    min_value: Real | None = None,
    max_value: Real | None = None,
    inclusive: bool = True,
) -> None:  # 校验数值范围。
    """确保数值落在给定区间里。

    Args:
        value: 要检查的数值，必须是 numbers.Real 的实例（bool 和 complex 不算数值）。
        name (str): 值的名称，用于错误消息。
        min_value (Real | None): 区间下界，None 表示不检查下界。
        max_value (Real | None): 区间上界，None 表示不检查上界。
        inclusive (bool): 是否包含边界值，默认为 True。
            True 时使用 >= / <=，False 时使用 > / <。

    Raises:
        ValueError: 当 value 为 None 或越界时抛出。
        TypeError: 当 value 不是数值类型（或为 bool）时抛出。
    """
    require_not_none(value, name)  # 先排除 None。
    if not is_real(value):  # 必须是实数（排除 bool 和 complex）。
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")  # 不是数值就报错。
    # 将值转为 float 做有限性检查，兼容 Fraction 等不被 math.isfinite 直接支持的 Real 子类型。
    try:
        _value_f = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(_value_f):  # NaN 和 inf 不允许通过范围检查。
        raise ValueError(f"{name} must be finite, got {value}")  # 非有限值直接报错。
    if min_value is not None:
        if not is_real(min_value):  # 必须是实数边界。
            raise TypeError(f"{name}: min_value must be numeric, got {type(min_value).__name__}")
        # 第 7 轮审查 MEDIUM-2 修复：math.isfinite(int) 对极大 int 抛 OverflowError，需先转 float 捕获。
        try:
            _min_f = float(min_value)
        except OverflowError as exc:
            raise ValueError(f"{name}: min_value int value too large to convert to float") from exc
        if not math.isfinite(_min_f):  # 下界必须是有限数。
            raise ValueError(f"{name}: min_value must be finite, got {min_value}")  # 非有限下界直接报错。
    if max_value is not None:
        if not is_real(max_value):  # 必须是实数边界。
            raise TypeError(f"{name}: max_value must be numeric, got {type(max_value).__name__}")
        # 第 7 轮审查 MEDIUM-2 修复：同 min_value。
        try:
            _max_f = float(max_value)
        except OverflowError as exc:
            raise ValueError(f"{name}: max_value int value too large to convert to float") from exc
        if not math.isfinite(_max_f):  # 上界必须是有限数。
            raise ValueError(f"{name}: max_value must be finite, got {max_value}")  # 非有限上界直接报错。
    if min_value is not None and max_value is not None and min_value > max_value:  # 配置错误应尽早暴露。
        raise ValueError(f"{name}: min_value ({min_value}) must not exceed max_value ({max_value})")  # 区间无效就报错。
    if min_value is not None:  # 如果给了下界。
        if inclusive and value < min_value:  # 包含边界时不能小于下界。
            raise ValueError(f"{name} must be >= {min_value}, got {value}")  # 越界就报错。
        if not inclusive and value <= min_value:  # 不包含边界时必须严格大于下界。
            raise ValueError(f"{name} must be > {min_value}, got {value}")  # 越界就报错。
    if max_value is not None:  # 如果给了上界。
        if inclusive and value > max_value:  # 包含边界时不能大于上界。
            raise ValueError(f"{name} must be <= {max_value}, got {value}")  # 越界就报错。
        if not inclusive and value >= max_value:  # 不包含边界时必须严格小于上界。
            raise ValueError(f"{name} must be < {max_value}, got {value}")  # 越界就报错。


def require_shape(value: Any, expected_shape: Sequence[int], *, name: str = "value") -> None:  # 校验对象形状。
    """确保对象形状与预期一致。

    优先读取对象的 shape 属性；如果没有 shape 属性但是一维序列
    （非 str/bytes），则用 (len(value),) 作为形状。

    Args:
        value: 要检查的对象，应有 shape 属性或为一维序列。
        expected_shape (Sequence[int]): 期望的形状，如 (3,) 或 (2, 4)。
        name (str): 对象的名称，用于错误消息，默认为 "value"。

    Raises:
        ValueError: 当 value 为 None 或形状不匹配时抛出。
        TypeError: 当 value 既没有 shape 属性也不是序列时抛出。
    """
    require_not_none(value, name)  # 先排除 None。
    shape = getattr(value, "shape", None)  # 优先看对象是否暴露 shape。
    if shape is None:  # 没有 shape 属性时。
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):  # 序列至少可以推断一维长度。
            shape = (len(value),)  # 把一维长度当作形状。
        else:  # 连序列都不是就没法判断。
            raise TypeError(f"{name} does not expose a shape, got {type(value).__name__}")  # 直接报错。
    if tuple(shape) != tuple(expected_shape):  # 实际形状和预期形状不一样。
        raise ValueError(f"{name} has shape {tuple(shape)}, expected {tuple(expected_shape)}")  # 报出形状不匹配。


def require_iterable(value: Any, *, name: str = "value") -> None:  # 校验对象是可迭代的。
    """确保对象是可迭代的，但字符串和字节串不算。

    Args:
        value: 要检查的对象。
        name (str): 对象的名称，用于错误消息，默认为 "value"。

    Raises:
        ValueError: 当 value 为 None 时抛出。
        TypeError: 当 value 是 str/bytes 或不是 Iterable 时抛出。

    Note:
        dict（Mapping）可通过此校验，因为 dict 是 Iterable（迭代键）。
        若需排除 dict，调用者应额外做 isinstance(value, Mapping) 检查。
        generator 可通过此校验，但 generator 只能遍历一次，
        调用者应立即 materialize（如 list()）以避免后续遍历失败。
    """
    require_not_none(value, name)  # 先排除 None。
    if isinstance(value, (str, bytes)):  # 字符串和字节串虽可迭代，但不是此处期望的容器类型。
        raise TypeError(f"{name} must be a non-string iterable, got {type(value).__name__}")  # 明确指出排除原因。
    if not isinstance(value, Iterable):  # 其余不可迭代对象直接拒绝。
        raise TypeError(f"{name} must be iterable, got {type(value).__name__}")  # 不满足条件就报错。


def coerce_finite_scalar(
    value: Any,
    *,
    name: str,
    min_value: Real | None = None,
    max_value: Real | None = None,
    inclusive: bool = True,
) -> float:
    """将输入值强制转换为有限浮点数，非数值或非有限值报错。

    统一校验+转换入口，供场景模块、传感器模块和网络模块复用，
    避免各模块各自定义同名的私有 _coerce_finite_scalar 函数。

    支持 Python 数值、NumPy 标量和 PyTorch 张量（单元素）。
    对 PyTorch 张量先调用 ``.detach().reshape(()).item()`` 取出 Python 数值，
    不引入对 torch 的硬依赖（延迟导入）。

    校验标准与 require_in_range 对齐：
    - 使用 is_real() 判断数值类型（排除 bool）
    - 检查值和边界值的有限性
    - 边界值类型必须为 Real
    - 支持 inclusive 参数控制开闭区间
    - 错误消息格式保持一致

    Args:
        value: 待转换的值，必须是数值类型（int、float 等）或
            单元素 PyTorch 张量，不接受 bool。
        name: 字段名，用于错误消息。
        min_value: 可选下界，如果提供则返回值必须 >= min_value（或 >）。
        max_value: 可选上界，如果提供则返回值必须 <= max_value（或 <）。
        inclusive: 是否包含边界值，默认为 True。
            True 时使用 >= / <=，False 时使用 > / <。

    Returns:
        有限的 float。

    Raises:
        TypeError: 值不是数值类型（bool 也不算），或边界值类型不正确。
        ValueError: 值是 inf 或 nan，边界值非有限，区间无效，或值超出范围。
    """
    require_not_none(value, name)
    # 热路径快通道：纯 Python float/int 直接走快速校验，跳过 torch 导入和
    # is_real/isinstance 开销。bool 是 int 的子类，需先排除。
    # 此优化在仿真物化热路径上把 coerce_finite_scalar 单次开销从 ~3μs 降到 ~0.3μs，
    # 对 50K 行 GT × 多次调用的场景有数量级提升。
    # 注意：int 也必须返回 float 以保持函数合同（Returns: 有限的 float）。
    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return _coerce_finite_scalar_check_bounds(
            value, name=name, min_value=min_value, max_value=max_value, inclusive=inclusive,
        )
    if value_type is int and not isinstance(value, bool):
        # 第 5 轮审查 MEDIUM-1 修复：删除原 L409-410 的 `math.isfinite(value)` 检查。
        # 原因：`math.isfinite(int)` 对极大 int（如 10**400）会先内部转 float，
        # 抛 OverflowError 而非返回 True/False，且该 OverflowError 不被下方
        # try/except 捕获（原 try/except 只包裹 float(value)），违反函数合同
        # （Raises: ValueError）。此外 L410 是死代码——math.isfinite(int) 对
        # 普通 int 永远返回 True，对极大 int 抛 OverflowError，永远不会返回 False。
        # 修复后：极大 int 的 OverflowError 由下方 try/except 统一捕获并转为 ValueError。
        # 极大 int（如 10**400）转 float 会 OverflowError，需捕获并转为 ValueError
        # 以保持函数合同（Raises: ValueError for 非有限值）。
        # 错误消息不含 {value}：极大 int 的 str() 可能长达数千字符，
        # 会污染日志和异常消息可读性。
        try:
            float_value = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} int value too large to convert to float") from exc
        return _coerce_finite_scalar_check_bounds(
            float_value, name=name, min_value=min_value, max_value=max_value, inclusive=inclusive,
        )
    # PyTorch 张量路径：延迟导入，不引入硬依赖。
    try:
        import torch
        if torch.is_tensor(value):
            if value.dtype == torch.bool:  # 布尔张量不是数值，与 is_bool_like 排除 bool 的口径对齐。
                raise TypeError(f"{name} must be numeric, got bool tensor")
            if value.numel() != 1:
                raise TypeError(f"{name} must be scalar-shaped, got tensor with shape {tuple(value.shape)}")
            try:
                scalar = float(value.detach().reshape(()).item())
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite numeric value") from exc
            if not math.isfinite(scalar):
                raise ValueError(f"{name} must be finite, got {scalar}")
            # 跳过下面的 is_real 检查，直接进入边界校验。
            return _coerce_finite_scalar_check_bounds(
                scalar, name=name, min_value=min_value, max_value=max_value, inclusive=inclusive,
            )
    except ImportError:
        pass
    if not is_real(value):  # 布尔值和非数值类型拒绝。
        # 尝试通过 .item() 方法取值（兼容 numpy 标量等）。
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                item_value = item_method()
            except (TypeError, ValueError, AttributeError) as exc:
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}") from exc
            if is_real(item_value):
                try:
                    scalar = float(item_value)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a finite numeric value") from exc
                if not math.isfinite(scalar):
                    raise ValueError(f"{name} must be finite, got {scalar}")
                return _coerce_finite_scalar_check_bounds(
                    scalar, name=name, min_value=min_value, max_value=max_value, inclusive=inclusive,
                )
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    try:
        scalar = float(value)  # 统一转成 float。
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(scalar):  # inf 和 nan 都会破坏后续计算。
        raise ValueError(f"{name} must be finite, got {scalar}")
    return _coerce_finite_scalar_check_bounds(
        scalar, name=name, min_value=min_value, max_value=max_value, inclusive=inclusive,
    )


def _coerce_finite_scalar_check_bounds(
    scalar: float,
    *,
    name: str,
    min_value: Real | None,
    max_value: Real | None,
    inclusive: bool = True,
) -> float:
    """边界校验子程序，供 coerce_finite_scalar 内部复用。"""
    # 第 7 轮审查 MEDIUM-2 修复：边界值若为极大 int（如 10**400），
    # math.isfinite(int) 内部转 float 会抛 OverflowError 而非返回 True/False，
    # 违反外层函数 Raises 契约（应抛 ValueError）。统一用 try/except 捕获并转为 ValueError。
    if min_value is not None:
        if not is_real(min_value):
            raise TypeError(f"{name}: min_value must be numeric, got {type(min_value).__name__}")
        try:
            _min_f = float(min_value)
        except OverflowError as exc:
            raise ValueError(f"{name}: min_value int value too large to convert to float") from exc
        if not math.isfinite(_min_f):
            raise ValueError(f"{name}: min_value must be finite, got {min_value}")
    if max_value is not None:
        if not is_real(max_value):
            raise TypeError(f"{name}: max_value must be numeric, got {type(max_value).__name__}")
        try:
            _max_f = float(max_value)
        except OverflowError as exc:
            raise ValueError(f"{name}: max_value int value too large to convert to float") from exc
        if not math.isfinite(_max_f):
            raise ValueError(f"{name}: max_value must be finite, got {max_value}")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{name}: min_value ({min_value}) must not exceed max_value ({max_value})")
    if min_value is not None:
        if inclusive and scalar < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {scalar}")
        if not inclusive and scalar <= min_value:
            raise ValueError(f"{name} must be > {min_value}, got {scalar}")
    if max_value is not None:
        if inclusive and scalar > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {scalar}")
        if not inclusive and scalar >= max_value:
            raise ValueError(f"{name} must be < {max_value}, got {scalar}")
    return scalar


def quality_below_floor(quality: float, floor: float, epsilon: float | None = None) -> bool:
    """判断质量值是否低于门槛，留浮点容差避免边界抖动。

    当 ``quality + epsilon < floor`` 时判定为低于门槛。
    quality 和 floor 必须为有限浮点数，否则抛出 ValueError。
    bool 类型会被拒绝（抛出 TypeError），因为布尔值与连续质量分数语义冲突。

    Args:
        quality: 质量值，必须为有限浮点数，不接受 bool。
        floor: 门槛值，必须为有限浮点数，不接受 bool。
        epsilon: 浮点容差，默认为 None 时使用 constants.QUALITY_FLOOR_EPSILON (1e-9)。
            显式传入 epsilon 可覆盖默认值；传 None 则从常量模块延迟导入。

    Returns:
        True 表示质量低于门槛。

    Raises:
        TypeError: 当 quality、floor 或 epsilon 为 bool 类型时抛出。
        ValueError: 当 quality 或 floor 为 NaN/inf 时抛出。
    """
    if is_bool_like(quality):  # bool 会隐式转成 0/1，与连续质量分数语义冲突。
        raise TypeError(f"quality must be numeric, got bool: {quality!r}")
    if is_bool_like(floor):  # floor 同样不接受 bool。
        raise TypeError(f"floor must be numeric, got bool: {floor!r}")
    if epsilon is not None and is_bool_like(epsilon):  # 显式传入的 epsilon 也不接受 bool。
        raise TypeError(f"epsilon must be numeric, got bool: {epsilon!r}")
    if epsilon is None:
        from liquidloc.common.constants import QUALITY_FLOOR_EPSILON
        epsilon = QUALITY_FLOOR_EPSILON
    # 第 7 轮审查 MEDIUM-1 修复：极大 int（如 10**400）转 float 会抛 OverflowError，
    # 违反函数合同（Raises: ValueError for NaN/inf）。统一用 try/except 捕获并转为 ValueError。
    try:
        q = float(quality)
        f = float(floor)
        e = float(epsilon)
    except OverflowError as exc:
        raise ValueError(
            "quality_below_floor received a value too large to convert to float"
        ) from exc
    if not (math.isfinite(q) and math.isfinite(f) and math.isfinite(e)):
        raise ValueError(
            f"quality_below_floor requires finite values, "
            f"got quality={quality!r}, floor={floor!r}, epsilon={epsilon!r}"
        )
    return q + e < f


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """去重但保留原始顺序。

    注意：重复元素的出现次数信息会丢失，仅保留首次出现。
    该函数是幂等的：对无重复列表恒等，对有重复列表 f(f(x))=f(x)≠x。
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def sample_axis_interval(
    value: Any,
    rng: Any = None,
    *,
    name: str,
) -> float:
    """从轴参数值中采样。

    支持两种形式：
      - 标量单值：直接返回 float(value)
      - 区间 [low, high]：在区间内均匀采样 float，并保留合理小数位

    保留位数策略（基于参数物理量级）：
      - 小数位 <= 1（概率类 0.0-1.0）：保留 2 位
      - 小数位 1-2（米级 0.5-6m）：保留 1 位
      - 小数位 >= 3（大尺度秒/PPM）：保留 0 位（整数秒/整数PPM）

    Args:
        value: 轴参数值，标量或 [low, high] 区间。
        rng: numpy 随机数生成器。
        name: 字段名，用于错误消息。

    Returns:
        均匀采样的 float，保留合理小数位。

    Raises:
        TypeError: value 不是数值类型或区间。
        ValueError: 区间下界 > 上界。
    """
    import hashlib
    import math
    import random as _rnd_stdlib

    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo = float(value[0])
        hi = float(value[1])
        if lo > hi:
            raise ValueError(f"{name} interval lower bound must be <= upper bound, got [{lo}, {hi}]")
        # rng=None 时用 hash 派生确定性 RNG（同区间下界策略，以 hash 为唯一随机源）
        if rng is None:
            _seed = int(hashlib.sha256(f"{name}:interval".encode("utf-8")).hexdigest()[:8], 16)
            rng = _rnd_stdlib.Random(_seed)

        # 规则：找档位数 ∈ [5, 20] 的最小精度 N。
        # 找不到时（区间宽度 > 20/10^N）：选最近 ≤ 20 档的精度。
        # 极端（width=2m, N=1 → 21 档）：退而求其次选 N=0（11 档）。
        raw_decimals_lo = len(str(lo).rstrip("0").split(".")[-1]) if "." in str(lo) else 0
        raw_decimals_hi = len(str(hi).rstrip("0").split(".")[-1]) if "." in str(hi) else 0
        start_decimals = max(0, min(raw_decimals_lo, raw_decimals_hi))
        best_decimals = start_decimals
        for N in range(start_decimals, 5):
            step_count = int(round((hi - lo) * (10 ** N))) + 1
            if 5 <= step_count <= 20:
                best_decimals = N; break
        else:
            # 找不到 [5, 20] → 选最近 ≤ 20 的精度（递减找）
            for N in range(start_decimals, -1, -1):
                sc = int(round((hi - lo) * (10 ** N))) + 1
                if sc <= 20:
                    best_decimals = N; break
            else:
                best_decimals = 0  # 极端兜底
        # 整数档位采样（避免浮点均匀分布在窄区间的精度退化问题）：
        #   step = 10^(-best_decimals)；档位数 N = int((hi-lo)/step) + 1 ∈ [1, 20]
        #   在 [0, N-1] 范围内均匀抽取整数档位索引，再转换为物理值
        step = 10.0 ** (-best_decimals)
        n_buckets = int(round((hi - lo) / step)) + 1  # 通常 ∈ [5, 20]
        bucket_idx = rng.randint(0, n_buckets - 1)
        snapped = round(lo + bucket_idx * step, best_decimals)
        sampled = max(lo, min(hi, snapped))
        return sampled  # ← 区间采样：直接返回离散档位值，跳过末尾通用 rounding
    elif is_real(value):
        sampled = float(value)
    else:
        raise TypeError(
            f"{name} must be a real number or a 2-value interval [low, high], got {type(value).__name__}"
        )

    # 标量路径：保留合理小数位（基于物理量级）
    abs_val = abs(sampled)
    if abs_val <= 1.0:
        decimals = 2
    elif abs_val < 10.0:
        decimals = 1
    else:
        decimals = 0
    return round(sampled, decimals)

