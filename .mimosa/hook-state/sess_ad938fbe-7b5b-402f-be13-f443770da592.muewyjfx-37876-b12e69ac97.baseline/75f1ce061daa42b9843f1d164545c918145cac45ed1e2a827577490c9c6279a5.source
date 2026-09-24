"""MILUV 原始表读取器，供准备流程和公开烟雾流程使用。

这个模块负责把 MILUV 原始表读成上层准备流程和烟雾流程能直接用的结构。

上游依赖:
- 文件系统上的 MILUV 序列目录（包含 imu.json, uwb.json, vio.json, gt.json）
- 官方实验表（config/experiments.csv）和锚点配置（config/uwb/anchors.yaml）

下游调用者:
- 准备流程脚本（通过 dataio/__init__.py 的 read_miluv_sequence 入口调用）
- 公开基准烟雾测试脚本

核心变量:
- bundle: 按内部约定组织的 MILUV 原始包字典
- read_report: 读取摘要报告

关键设计决策:
- 优先使用本地 anchor_layout.json，不存在时回退到官方元数据
- 时间戳字段统一按浮点类型处理
- 锚点布局支持本地 JSON 和官方元数据两种来源
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.constants import MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE  # 官方锚点元数据来源标识（单源真相，D9 漂移根因修复）。
from liquidloc.common.io_utils import loads_json_text, read_json
from liquidloc.common.validation import is_string_like


_REQUIRED_STREAM_FILES = {  # MILUV 序列默认必须具备的流文件。
    'imu': 'imu.json',  # IMU 流文件名。
    'uwb': 'uwb.json',  # UWB 流文件名。
    'vio': 'vio.json',  # VIO 流文件名。
    'gt': 'gt.json',  # 真值流文件名。
}  # 必需流文件映射结束。


def _read_json_records(path: Path) -> list[dict[str, Any]]:  # 读取必须存在的 JSON 记录文件。
    """读取必须存在的 JSON 记录文件，并要求内容是字典列表。"""  # 这里直接校验原始文件形状，避免后面消费方再猜类型。
    if not path.is_file():  # 文件不存在就报错。
        raise FileNotFoundError(f"Required data file not found: {path}")  # 文件缺失时直接中止。
    payload = read_json(path)  # 按严格标准 JSON 语义解析。
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):  # 内容必须是字典列表。
        raise TypeError(f"Expected a list[dict] payload in {path}")  # 内容形状不对就报错。
    return payload  # 返回记录列表。


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:  # 读取可选 JSON 对象文件；不存在或无效就返回 None。
    """读取可选的 JSON 对象文件；不存在或内容无效就返回 None。"""  # 可选元数据缺失或损坏时不报错，统一返回 None。
    if not path.is_file():  # 没有文件就视为没有这个可选元数据。
        return None  # 可选文件缺失不算错误。
    try:
        payload = read_json(path)  # 用严格标准 JSON 语义读取可选对象。
    except ValueError:  # JSON 语法损坏或包含非有限值时视为无法获取。
        return None
    if not isinstance(payload, dict):  # 可选文件也必须是字典。
        return None  # 非字典内容同样视为无法获取。
    return payload  # 返回对象。


def _read_csv_rows(path: Path) -> list[dict[str, str]]:  # 读取必须存在的 CSV 文件。
    """读取必须存在的 CSV 文件，并按字典行返回。"""  # 官方实验表是后续找锚点布局的入口，所以必须能读成字典行。
    if not path.is_file():  # 文件不存在就报错。
        raise FileNotFoundError(f"Required data file not found: {path}")  # CSV 文件缺失时直接中止。
    with path.open('r', encoding='utf-8', newline='') as handle:  # 用 UTF-8 和空换行模式读取 CSV。
        return list(csv.DictReader(handle))  # 直接把 CSV 转成字典行列表。


def _find_official_metadata_paths(raw_root: Path) -> tuple[Path, Path] | None:  # 查找官方实验表和锚点配置。
    """在原始根目录或其父目录下查找官方实验表和锚点配置。"""  # 这里兼容原始根目录和父目录两种摆放方式，减少用户目录组织差异带来的失败。
    for candidate_root in (raw_root, raw_root.parent):  # 先看根目录，再看父目录。
        experiments_path = candidate_root / 'config' / 'experiments.csv'  # 官方实验表路径。
        anchors_path = candidate_root / 'config' / 'uwb' / 'anchors.yaml'  # 官方锚点配置路径。
        if experiments_path.is_file() and anchors_path.is_file():  # 两个文件都存在才算找到。
            return experiments_path, anchors_path  # 返回这对官方路径。
    return None  # 没找到就返回空。


def _dedupe_preserve_order(items: list[str]) -> list[str]:  # 去重但保留顺序。
    """去重但保留原始顺序。"""  # 只去掉重复项，不打乱原始顺序。
    seen: set[str] = set()  # 已见过的条目。
    ordered: list[str] = []  # 去重后的顺序结果。
    for item in items:  # 逐个看输入项。
        if item not in seen:  # 第一次出现才保留。
            seen.add(item)  # 标记为已见过。
            ordered.append(item)  # 加入结果。
    return ordered  # 返回顺序保留的去重列表。


def _list_miluv_sequence_dirs(raw_root: Path) -> list[Path]:  # 列出可处理的 MILUV 序列目录。
    """列出可处理的 MILUV 序列目录，排除 config 和隐藏目录。"""  # 委托给共享函数，保证过滤规则一致。
    from liquidloc.dataio.readers import list_sequence_dirs  # 延迟导入避免循环依赖。
    return list_sequence_dirs(raw_root)


def _strip_wrapping_quotes(value: Any) -> Any:  # 去掉字符串两端多余引号。
    """去掉字符串两端包裹的单引号或双引号。"""  # 官方配置里有时会把字段值写成带引号字符串，这里先统一清理掉外层引号。
    if not is_string_like(value):  # 非字符串直接原样返回。
        return value  # 只有字符串才需要处理引号。
    text = str(value).strip()  # 先去掉首尾空白。
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:  # 两端引号一致时去掉外层引号。
        return text[1:-1]  # 去掉包裹引号。
    return text  # 否则返回去空白后的文本。


def _sort_anchor_id(value: Any) -> tuple[int, Any]:  # 给锚点编号提供稳定排序键。
    """给锚点编号提供稳定排序键，数字优先，文本次之。"""  # 锚点编号可能混有数字和文本，所以先把顺序规则固定下来。
    text = str(_strip_wrapping_quotes(value))  # 先清理引号再转字符串。
    if text.isdigit():  # 如果是纯数字，就按整数排。
        return 0, int(text)  # 数字放在前面排序。
    return 1, text  # 否则按文本排。


def _normalize_anchor_id(anchor_id: Any) -> Any:  # 把锚点编号统一成更稳定的表示形式。
    """把锚点编号统一成更稳定的表示形式。"""  # 统一编号表示后，后续比较和排序就不会被原始写法扰乱。
    token = _strip_wrapping_quotes(anchor_id)  # 先去掉包裹引号。
    if is_string_like(token) and str(token).isdigit():  # 数字字符串转整数。
        return int(token)  # 数字字符串统一转整数。
    return token  # 其他值原样返回。


def _normalize_anchor_position(position: Any) -> list[float] | None:  # 把锚点位置统一成浮点列表。
    """把锚点位置统一成浮点列表；失败就返回 None。"""  # 位置字段可能是字符串、元组或其他容器，先标准化再判断维度。
    payload = position  # 先把输入放进临时变量。
    if is_string_like(payload):  # 如果是字符串，先尝试按 JSON 解析。
        try:  # 先尝试把字符串形式的位置解析成 JSON 对象。
            payload = loads_json_text(str(payload))  # 字符串形式的坐标数组先转成对象。
        except ValueError:  # JSON 解析失败或出现非有限值时直接判定无效。
            return None  # 解析失败就返回空。
    if isinstance(payload, (str, bytes)):  # 纯字符串或 bytes 不适合作为坐标容器。
        return None  # 纯字符串或 bytes 不能作为坐标序列。
    try:  # 再把已经是容器的坐标统一转成浮点列表。
        result = [float(coord) for coord in payload]  # 把每个坐标都转成浮点数。
    except (TypeError, ValueError):  # 任何转换失败都说明这个位置字段不可用。
        return None  # 任何转换失败都返回空。
    if not result:  # 空坐标列表语义无效，等同于解析失败。
        return None  # 空坐标列表也返回空。
    return result  # 返回有效坐标列表。


def _assess_anchor_layout_candidate(  # 评估一个候选锚点布局，并返回规范化结果、原始元数据和审核报告。
    anchor_layout: dict[str, Any],  # 候选布局字典，通常来自本地文件或官方元数据。
    *,  # 后面的参数只允许使用关键字传入，避免调用时把来源和路径位置弄混。
    source: str,  # 候选布局的来源标识，用于报告里说明数据从哪里来。
    source_paths: list[str],  # 候选布局来自哪些文件路径，用于追溯和调试。
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:  # 返回规范化布局、原始元数据和报告三元组。
    """评估一个锚点布局候选项是否满足教师投影所需的基础条件。"""  # 这里先判断最小合同，再决定是否能继续支持教师投影。
    metadata = dict(anchor_layout)  # 复制一份，避免直接改调用方对象。
    metadata.setdefault('source', source)  # 如果没有来源字段，就补上来源标识。
    metadata['source_paths'] = list(source_paths)  # 把来源路径列表写进去，方便回溯。

    blockers: list[str] = []  # 收集阻塞条件。
    normalized_anchor_ids: list[Any] | None = None  # 标准化后的锚点编号列表。
    normalized_anchor_positions: list[list[float]] | None = None  # 标准化后的锚点位置列表。
    position_dim: int | None = None  # 位置维度先留空。

    raw_anchor_ids = metadata.get('anchor_ids')  # 读取原始锚点编号。
    if raw_anchor_ids is None or isinstance(raw_anchor_ids, (str, bytes)):  # 缺失或类型不对都算阻塞。
        blockers.append('anchor_ids_missing_or_invalid')  # 记录锚点编号问题。
    else:  # 否则就继续标准化锚点编号。
        try:  # 先把候选编号列表转成稳定结构。
            normalized_anchor_ids = [_normalize_anchor_id(anchor_id) for anchor_id in list(raw_anchor_ids)]  # 逐个标准化锚点编号。
        except TypeError:  # 不是可迭代对象时直接视为无效。
            blockers.append('anchor_ids_missing_or_invalid')  # 容器不可迭代就记阻塞。
        else:  # 只有容器可迭代时才继续。
            if not normalized_anchor_ids:  # 空列表也不行。
                blockers.append('anchor_ids_missing_or_invalid')  # 空列表也说明锚点编号不可用。
            elif any(aid is None for aid in normalized_anchor_ids):  # 含 None 元素也不行。
                blockers.append('anchor_ids_missing_or_invalid')  # None 锚点编号视为无效。
                normalized_anchor_ids = None  # 清空以阻止后续长度比较。
            else:  # 只有编号能正常转成字符串后才继续标记。
                metadata['anchor_ids'] = normalized_anchor_ids  # 把标准化结果写回元数据。

    raw_anchor_positions = metadata.get('anchor_positions')  # 读取原始锚点位置。
    if raw_anchor_positions is None or isinstance(raw_anchor_positions, (str, bytes)):  # 缺失或类型不对都算阻塞。
        blockers.append('anchor_positions_missing_or_invalid')  # 缺失或类型不对都算锚点位置无效。
    else:  # 否则继续处理锚点位置。
        try:  # 再尝试把位置列表转成可检查对象。
            raw_positions = list(raw_anchor_positions)  # 尝试把位置容器转成列表。
        except TypeError:  # 位置字段不可迭代时直接判坏。
            blockers.append('anchor_positions_missing_or_invalid')  # 位置字段不可迭代时记阻塞。
            raw_positions = []  # 这时没有任何可继续解析的位置。

        normalized_positions: list[list[float]] = []  # 标准化后的坐标列表。
        dims: set[int] = set()  # 记录每个坐标的维度。
        for raw_position in raw_positions:  # 逐个位置检查。
            coords = _normalize_anchor_position(raw_position)  # 把单个位置标准化成浮点列表。
            if coords is None:  # 解析失败就直接记阻塞并清空结果。
                blockers.append('anchor_positions_missing_or_invalid')  # 单个位置解析失败也算无效。
                normalized_positions = []  # 既然出现坏项，就清空已收集结果。
                dims = set()  # 维度集合也清空，避免误判。
                break  # 一旦维度混杂，就停止继续判定。
            dims.add(len(coords))  # 记录这个坐标的维度。
            normalized_positions.append(coords)  # 保存标准化坐标。

        if not normalized_positions:  # 没有任何有效坐标也算阻塞。
            if 'anchor_positions_missing_or_invalid' not in blockers:  # 没有基础无效标记时先补上。
                blockers.append('anchor_positions_missing_or_invalid')  # 没有任何有效坐标时补上阻塞标记。
        elif len(dims) != 1:  # 坐标维度不一致也不行。
            blockers.append('anchor_positions_mixed_dimensions')  # 维度混杂说明布局结构不统一。
        else:  # 所有坐标都有效且维度一致时才进入这里。
            position_dim = dims.pop()  # 取出统一的维度。
            normalized_anchor_positions = normalized_positions  # 保存标准化结果。
            metadata['anchor_positions'] = normalized_positions  # 写回元数据。
            metadata['position_dim'] = position_dim  # 记录位置维度。

    if normalized_anchor_ids is not None and normalized_anchor_positions is not None:  # 两边都解析成功时再比较长度。
        if len(normalized_anchor_ids) != len(normalized_anchor_positions):  # 编号数和位置数必须一致。
            blockers.append('anchor_ids_anchor_positions_length_mismatch')  # 长度不一致就记阻塞。

    if not blockers and position_dim != 2:  # 没有别的问题但维度不是 2 时，教师投影仍然不满足。
        blockers.append(f'anchor_positions_dim_{position_dim}_requires_2d_teacher')  # 记录需要二维教师的原因。

    teacher_ready = not blockers and position_dim == 2  # 只有没有阻塞且是二维时才算教师就绪。
    report = {  # 组装报告，给上层看。
        'anchor_layout_metadata_available': True,  # 说明确实拿到了元数据。
        'anchor_layout_metadata_source': source,  # 记录来源。
        'anchor_layout_source_paths': list(source_paths),  # 记录来源路径。
        'anchor_layout_position_dim': position_dim,  # 记录位置维度。
        'anchor_layout_teacher_ready': teacher_ready,  # 记录教师投影是否可用。
        'anchor_layout_blockers': blockers,  # 记录所有阻塞条件。
    }  # 锚点布局评估报告结束。
    if len(source_paths) == 1 and Path(source_paths[0]).name == 'anchor_layout.json':  # 如果来源就是单个布局文件，就补一个直观路径。
        report['anchor_layout_path'] = source_paths[0]  # 单一路径来源时补一个直观路径字段。

    if teacher_ready:  # 就绪时返回规范化布局，否则返回 None 和报告。
        result = dict(metadata)  # 先做浅拷贝。
        if isinstance(result.get('anchor_ids'), list):  # anchor_ids 是可变列表，需要独立副本。
            result['anchor_ids'] = list(result['anchor_ids'])  # 断开与 metadata 的共享引用。
        if isinstance(result.get('anchor_positions'), list):  # anchor_positions 是嵌套可变列表，需要深隔离。
            result['anchor_positions'] = [list(p) for p in result['anchor_positions']]  # 逐层断开共享引用。
        if isinstance(result.get('source_paths'), list):  # source_paths 也是可变列表。
            result['source_paths'] = list(result['source_paths'])  # 断开与 metadata 的共享引用。
        return result, metadata, report  # 返回独立副本、原始元数据和报告。
    return None, metadata, report  # 未就绪时只返回元数据和报告。


def _discover_official_anchor_layout_candidate(seq_id: str, raw_root: Path) -> tuple[dict[str, Any] | None, str | None, list[str]]:  # 从官方实验表和锚点配置里找对应序列的官方布局。
    """从官方实验表和锚点配置里发现某条序列对应的官方锚点布局。"""  # 这条链路只负责找候选，不负责最终裁决能不能用。
    metadata_paths = _find_official_metadata_paths(raw_root)  # 先找官方元数据文件。
    if metadata_paths is None:  # 没找到就直接返回空。
        return None, None, []  # 没有官方元数据路径。

    experiments_path, anchors_path = metadata_paths  # 拆出实验表和锚点配置路径。
    try:  # 官方实验表缺失或不可读时视为无法获取，与锚点 YAML 的回退语义一致。
        experiment_rows = _read_csv_rows(experiments_path)  # 读取官方实验表。
    except FileNotFoundError:  # TOCTOU：文件在 _find_official_metadata_paths 检查后被删除。
        return None, None, []  # 官方实验表不可用时返回空，不阻断本地数据的读取。
    experiment_row = next(  # 从实验表里找与 seq_id 匹配的那一行。
        (  # 生成一个只保留匹配实验行的惰性序列。
            row for row in experiment_rows  # 逐行扫描。
            if str(row.get('experiment') or '').strip() == seq_id  # experiment 字段必须和 seq_id 相等。
        ),  # next 的默认值放在这里，避免找不到匹配行时报错。
        None,  # 没有匹配到时返回空。
    )  # 按实验编号查找对应行。
    if experiment_row is None:  # 没找到对应实验行就返回空。
        return None, None, []  # 没有对应实验行。

    anchor_constellation = str(_strip_wrapping_quotes(experiment_row.get('anchor_constellation') or '')).strip()  # 取出并清理星座名。
    if not anchor_constellation:  # 星座名为空就没法继续。
        return None, None, []  # 没有星座名。

    try:  # 官方锚点 YAML 损坏时视为无法获取，与本地 JSON 损坏时的回退语义一致。
        anchors_cfg = load_yaml_config(anchors_path)  # 读取官方锚点 YAML。
    except (FileNotFoundError, ValueError, TypeError):  # YAML 缺失、语法损坏或顶层非映射都视为官方元数据不可用。
        return None, None, []  # 官方锚点配置不可用时返回空，不阻断本地数据的读取。
    constellation_payload = None  # 先不指定具体星座内容。
    for constellation_key, constellation_value in anchors_cfg.items():  # 在 YAML 里找匹配的星座键。
        if str(_strip_wrapping_quotes(constellation_key)) == anchor_constellation:  # 键名和实验表星座名一致时命中。
            constellation_payload = constellation_value  # 命中的星座配置先暂存。
            break  # 找到匹配布局后就不必继续扫了。
    if not isinstance(constellation_payload, dict) or not constellation_payload:  # 星座内容必须是非空字典。
        return None, None, []  # 星座内容无效。

    anchor_ids: list[Any] = []  # 收集锚点编号。
    anchor_positions: list[Any] = []  # 收集锚点位置。
    for anchor_id in sorted(constellation_payload, key=_sort_anchor_id):  # 按稳定顺序遍历锚点。
        anchor_ids.append(_normalize_anchor_id(anchor_id))  # 记录锚点编号。
        anchor_positions.append(constellation_payload[anchor_id])  # 记录锚点位置。

    return {  # 返回一个官方候选锚点布局。
        'anchor_ids': anchor_ids,  # 锚点编号列表。
        'anchor_positions': anchor_positions,  # 锚点位置列表。
        'layout_id': f'miluv_official_constellation_{anchor_constellation}',  # 布局编号。
        'experiment': seq_id,  # 对应实验编号。
        'anchor_constellation': anchor_constellation,  # 星座名。
        'source': MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE,  # 来源标识（单源常量，D9 漂移根因修复）。
    }, MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE, [str(experiments_path), str(anchors_path)]  # 同时返回来源标识和来源路径。


def _probe_required_streams(seq_dir: Path) -> tuple[dict[str, Any], list[str]]:  # 检查一条序列里必需流的存在性、行数和基础字段。
    """检查一条序列里必需流的存在性和基础完整性。"""  # 这里给上层一个最早的失败点，方便解释为什么不可用。
    stream_report: dict[str, Any] = {}  # 各流的检查报告放这里。
    missing_or_invalid: list[str] = []  # 收集缺失或无效的流名。
    for stream_name, filename in _REQUIRED_STREAM_FILES.items():  # 逐个检查必需流文件。
        stream_path = seq_dir / filename  # 拼出当前流文件的路径。
        stream_state = {  # 先构造基础状态。
            'path': str(stream_path),  # 文件路径字符串。
            'present': stream_path.is_file(),  # 文件是否存在。
            'row_count': None,  # 行数先留空。
            'non_empty': False,  # 是否非空先默认否。
        }  # 单流基础状态字典闭合。
        if not stream_state['present']:  # 文件不存在就记缺失。
            missing_or_invalid.append(stream_name)  # 记录这个流名。
            stream_report[stream_name] = stream_state  # 把状态写入报告。
            continue  # 继续检查下一个流。
        try:  # 尝试读取当前流文件。
            rows = _read_json_records(stream_path)  # 读取这个流的 JSON 行。
        except (FileNotFoundError, TypeError, ValueError) as exc:  # 读取或解析失败都算无效。
            stream_state['error'] = f'{type(exc).__name__}: {exc}'  # 记录具体错误信息。
            missing_or_invalid.append(stream_name)  # 这个流加入无效列表。
        else:  # 只有文件读成功时才会走这里。
            stream_state['row_count'] = len(rows)  # 记录行数。
            stream_state['non_empty'] = bool(rows)  # 记录是否非空。
            if not rows:  # 空流单独标记。
                stream_state['error'] = 'empty_stream'  # 记录空流错误。
                missing_or_invalid.append(stream_name)  # 空流也算无效。
        stream_report[stream_name] = stream_state  # 当前流检查完后写回总报告。
    return stream_report, missing_or_invalid  # 返回流报告和无效流列表。


def _assess_teacher_projection_audit(anchor_layout_metadata: dict[str, Any] | None) -> dict[str, Any]:  # 判断锚点布局是否能支撑教师投影到二维平面。
    """检查教师投影是否可用，并说明当前布局是否能投到二维。"""  # 这里区分“不需要投影”和“根本不能投影”。
    if not isinstance(anchor_layout_metadata, dict):  # 没有元数据就直接说明缺失。
        return {'status': 'missing_anchor_layout', 'supported': False}  # 没有元数据就直接说明缺失。

    position_dim = anchor_layout_metadata.get('position_dim')  # 读取位置维度。
    if position_dim == 2:  # 已经是二维时，不需要额外投影。
        return {  # 已经是二维时，不需要额外投影。
            'status': 'not_needed',  # 不需要投影。
            'supported': True,  # 这个路径是支持的。
            'teacher_anchor_position_dim': 2,  # 教师锚点维度就是二维。
        }  # 二维布局的快速返回结束。

    if (  # 只有官方来源且原始维度是三维时，才考虑投影路径。
        str(anchor_layout_metadata.get('source') or '') != MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE  # 只有官方来源才考虑投影路径（单源常量，D9 漂移根因修复）。
        or position_dim != 3  # 还必须是三维布局。
    ):  # 这是教师投影可用性的前置门槛。
        return {  # 这条路径不可用时直接返回。
            'status': 'projection_path_unavailable',  # 投影路径不可用。
            'supported': False,  # 当前不支持。
            'original_anchor_position_dim': position_dim,  # 原始维度写出来。
        }  # 投影路径不可用的报告结束。

    raw_anchor_ids = anchor_layout_metadata.get('anchor_ids')  # 取锚点编号。
    raw_anchor_positions = anchor_layout_metadata.get('anchor_positions')  # 取锚点位置。
    if isinstance(raw_anchor_ids, (str, bytes)) or isinstance(raw_anchor_positions, (str, bytes)):  # 字符串形式不接受。
        return {  # 字符串形式的输入直接视为无效。
            'status': 'projection_inputs_invalid',  # 输入不合法。
            'supported': False,  # 这条路径不支持。
            'original_anchor_position_dim': 3,  # 原始布局仍按三维记录。
        }  # 字符串输入无效的报告结束。

    try:  # 先把原始锚点位置列表标准化成可检查的坐标容器。
        anchor_ids = list(raw_anchor_ids or [])  # 转成列表，方便后面比较长度。
        anchor_positions = list(raw_anchor_positions or [])  # 转成列表，方便后面比较长度。
    except TypeError:  # 如果根本无法迭代，就说明输入不合法。
        return {  # 输入不可迭代也不行。
            'status': 'projection_inputs_invalid',  # 输入不可迭代也不行。
            'supported': False,  # 这条路径不支持。
            'original_anchor_position_dim': 3,  # 原始布局仍按三维记录。
        }  # 不可迭代输入的报告结束。

    if not anchor_ids or len(anchor_ids) != len(anchor_positions):  # 编号数量和位置数量必须一致。
        return {  # 单个坐标分量数量不对时直接失败。
            'status': 'projection_inputs_invalid',  # 输入不合法。
            'supported': False,  # 这条路径不支持。
            'original_anchor_position_dim': 3,  # 原始布局仍按三维记录。
        }  # 维度不对的报告结束。

    for anchor_position in anchor_positions:  # 逐个检查每个三维坐标。
        if isinstance(anchor_position, (str, bytes)):  # 字符串不算坐标序列。
            return {  # 字符串输入直接判为无效。
                'status': 'projection_inputs_invalid',  # 输入不合法。
                'supported': False,  # 这条路径不支持。
                'original_anchor_position_dim': 3,  # 原始布局仍按三维记录。
            }  # 字符串输入无效的报告结束。
        try:  # 尝试把单个坐标转成可迭代列表。
            coords = list(anchor_position)  # 尝试把坐标转成列表。
        except TypeError:  # 单个坐标也无法迭代时直接失败。
            return {  # 单个坐标不可迭代，说明这个布局不能投影。
                'status': 'projection_inputs_invalid',  # 输入不合法。
                'supported': False,  # 这条路径不支持。
                'original_anchor_position_dim': 3,  # 原始布局仍按三维记录。
            }  # 单个坐标不可迭代的报告结束。
        if len(coords) != 3:  # 三维布局必须每个点都有 3 个分量。
            return {  # 单个坐标分量数量不对，直接失败返回。
                'status': 'projection_inputs_invalid',  # 输入不合法。
                'supported': False,  # 这条路径不支持。
                'original_anchor_position_dim': len(coords),  # 这里记录实际长度，方便排查。
            }  # 维度不对的报告结束。

    return {  # 汇总教师投影审计结果。
        'status': 'ready_xy_projection',  # 可以投影到 XY 平面。
        'supported': True,  # 这个路径支持。
        'original_anchor_position_dim': 3,  # 原始是三维。
        'teacher_anchor_position_dim': 2,  # 投影后给教师用的是二维。
        'projection': 'xy',  # 投影方式是 XY。
        'ignored_axis': 'z',  # 忽略 Z 轴。
    }  # 教师投影审计结束。


def _inspect_anchor_layout_readiness(seq_id: str, raw_root: Path) -> dict[str, Any]:  # 汇总本地或官方锚点布局的可用性判断。
    """检查序列锚点布局是否可用于教师投影或本地合同。"""  # 这一步把本地文件和官方元数据合并成一个总判断。
    seq_dir = raw_root / seq_id  # 拼出序列目录。
    anchor_layout_source = None  # 先不指定来源。
    source_paths: list[str] = []  # 来源路径列表先空着。
    anchor_layout_metadata = None  # 解析后的元数据先空着。
    anchor_layout_report = {  # 先准备一份默认报告，结构与 _assess_anchor_layout_candidate 返回值对齐。
        'anchor_layout_metadata_available': False,  # 先默认没有拿到元数据。
        'anchor_layout_metadata_source': None,  # 来源标识先空着。
        'anchor_layout_source_paths': [],  # 来源路径先空着。
        'anchor_layout_teacher_ready': False,  # 先默认不能给教师投影使用。
        'anchor_layout_blockers': [],  # 阻塞条件先空着。
        'anchor_layout_position_dim': None,  # 位置维度先未知。
    }  # 默认布局报告闭合。

    anchor_layout_path = seq_dir / 'anchor_layout.json'  # 本地布局文件路径。
    local_json_invalid = False  # 标记本地布局文件是否存在但 JSON 无效。
    if anchor_layout_path.is_file():  # 本地布局文件存在时先用本地合同。
        source_paths = [str(anchor_layout_path)]  # 记录本地来源路径。
        anchor_layout_candidate = _read_optional_json_object(anchor_layout_path)  # 读取本地布局 JSON；损坏时自动返回 None。
        if anchor_layout_candidate is not None:  # 本地 JSON 有效时才评估。
            anchor_layout_source = str(anchor_layout_candidate.get('source') or 'per_sequence_anchor_layout_json')  # 记录来源字段。
            _, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 评估本地布局候选项。
                anchor_layout_candidate,  # 本地布局候选项。
                source=anchor_layout_source,  # 本地来源标识。
                source_paths=source_paths,  # 本地来源路径。
            )  # 本地布局候选评估完成。
        else:  # 本地文件存在但 JSON 无效，标记后继续尝试官方元数据（与 read_miluv_sequence 行为一致）。
            local_json_invalid = True  # 记录本地 JSON 无效，供后续回退逻辑使用。

    if anchor_layout_metadata is None:  # 本地布局不可用时，转而看官方元数据。
        official_candidate, official_source, official_source_paths = _discover_official_anchor_layout_candidate(seq_id, raw_root)  # 尝试找官方元数据。
        if official_candidate is not None and official_source is not None:  # 找到官方候选时，改用官方来源。
            anchor_layout_source = official_source  # 记录改用的官方来源。
            source_paths = list(official_source_paths)  # 记录官方来源路径。
            _, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 评估官方候选项。
                official_candidate,  # 官方布局候选项。
                source=official_source,  # 官方来源标识。
                source_paths=source_paths,  # 官方来源路径。
            )  # 官方布局候选评估完成。
        elif local_json_invalid:  # 本地 JSON 无效且官方元数据也不可用，记录本地无效原因。
            anchor_layout_report = {  # 补充本地 JSON 无效的阻塞信息。
                'anchor_layout_metadata_available': False,  # 没有拿到元数据。
                'anchor_layout_teacher_ready': False,  # 不能给教师投影使用。
                'anchor_layout_blockers': ['anchor_layout_json_invalid'],  # 记录 JSON 无效。
                'anchor_layout_position_dim': None,  # 位置维度未知。
            }

    anchor_ids_available = bool(  # 只要有非空锚点编号列表，就认为可用。
        isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
        and isinstance(anchor_layout_metadata.get('anchor_ids'), list)  # 再确认锚点编号是列表。
        and anchor_layout_metadata.get('anchor_ids')  # 最后确认列表非空。
    )  # 锚点编号可用性判断结束。
    anchor_positions_available = bool(  # 只要有非空锚点位置列表，就认为可用。
        isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
        and isinstance(anchor_layout_metadata.get('anchor_positions'), list)  # 再确认锚点位置是列表。
        and anchor_layout_metadata.get('anchor_positions')  # 最后确认列表非空。
    )  # 锚点位置可用性判断结束。
    teacher_projection_audit = _assess_teacher_projection_audit(anchor_layout_metadata)  # 再看教师投影是否支持。
    ready = bool(  # 只有编号、位置和投影三者都满足，才算就绪。
        anchor_ids_available  # 编号可用。
        and anchor_positions_available  # 位置可用。
        and teacher_projection_audit.get('supported')  # 投影审计也支持。
    )  # ready 判定结束。
    return {  # 汇总本地或官方锚点布局的可用性。
        'ready': ready,  # 最终是否就绪。
        'source': anchor_layout_source,  # 使用的来源。
        'source_paths': source_paths,  # 来源路径。
        'metadata_available': bool(anchor_layout_metadata is not None),  # 是否有元数据。
        'anchor_ids_available': anchor_ids_available,  # 编号是否可用。
        'anchor_positions_available': anchor_positions_available,  # 位置是否可用。
        'position_dim': anchor_layout_report.get('anchor_layout_position_dim'),  # 位置维度。
        'blockers': list(anchor_layout_report.get('anchor_layout_blockers') or []),  # 阻塞条件。
        'teacher_projection_audit': teacher_projection_audit,  # 教师投影审计结果。
    }  # 可用性汇总结束。


def _inspect_local_anchor_layout_contract(seq_id: str, raw_root: Path) -> dict[str, Any]:  # 只看本地锚点布局文件是否满足最小合同。
    """检查本地锚点布局合同是否满足读取和教师投影要求。"""  # 这里只判断本地合同本身，不混入官方元数据的推断。
    seq_dir = raw_root / seq_id  # 拼出序列目录。
    anchor_layout_path = seq_dir / 'anchor_layout.json'  # 本地布局文件路径。
    if not anchor_layout_path.is_file():  # 没有布局文件就直接返回失败报告。
        return {  # 没有布局文件就直接返回失败报告。
            'ready': False,  # 没有布局文件就不可就绪。
            'source': None,  # 来源为空。
            'source_paths': [],  # 来源路径为空。
            'metadata_available': False,  # 没有元数据。
            'anchor_ids_available': False,  # 编号不可用。
            'anchor_positions_available': False,  # 位置不可用。
            'position_dim': None,  # 维度未知。
            'blockers': ['missing_anchor_layout_json'],  # 记录缺失布局文件。
        }  # 缺失布局文件的报告结束。

    source_paths = [str(anchor_layout_path)]  # 记录来源路径。
    anchor_layout_candidate = _read_optional_json_object(anchor_layout_path)  # 读取本地布局 JSON；损坏时自动返回 None。
    if anchor_layout_candidate is None:  # 文件存在但 JSON 无效，返回失败报告。
        return {  # 本地 JSON 坏掉时直接返回失败。
            'ready': False,  # 本地 JSON 解析失败时直接不可就绪。
            'source': 'per_sequence_anchor_layout_json',  # 失败来源仍记为本地布局文件。
            'source_paths': source_paths,  # 路径信息保留。
            'metadata_available': False,  # 没有可用元数据。
            'anchor_ids_available': False,  # 编号不可用。
            'anchor_positions_available': False,  # 位置不可用。
            'position_dim': None,  # 维度未知。
            'blockers': ['anchor_layout_json_invalid'],  # 记录 JSON 无效。
        }  # 本地 JSON 异常报告结束。

    anchor_layout, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 评估本地候选项。
        anchor_layout_candidate,  # 本地布局候选项。
        source=str(anchor_layout_candidate.get('source') or 'per_sequence_anchor_layout_json'),  # 来源字段。
        source_paths=source_paths,  # 来源路径。
    )  # 本地候选评估完成。
    return {  # 汇总本地锚点布局合同的判断结果。
        'ready': anchor_layout is not None,  # 只有候选被判定可用才算 ready。
        'source': str(anchor_layout_candidate.get('source') or 'per_sequence_anchor_layout_json'),  # 使用的来源。
        'source_paths': source_paths,  # 来源路径。
        'metadata_available': bool(anchor_layout_metadata is not None),  # 是否有元数据。
        'anchor_ids_available': bool(  # 是否有可用的锚点编号。
            isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
            and isinstance(anchor_layout_metadata.get('anchor_ids'), list)  # 再确认锚点编号是列表。
            and anchor_layout_metadata.get('anchor_ids')  # 最后确认列表非空。
        ),  # 锚点编号可用性判断结束。
        'anchor_positions_available': bool(  # 是否有可用的锚点位置。
            isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
            and isinstance(anchor_layout_metadata.get('anchor_positions'), list)  # 再确认锚点位置是列表。
            and anchor_layout_metadata.get('anchor_positions')  # 最后确认列表非空。
        ),  # 锚点位置可用性判断结束。
        'position_dim': anchor_layout_report.get('anchor_layout_position_dim'),  # 位置维度。
        'blockers': list(anchor_layout_report.get('anchor_layout_blockers') or []),  # 阻塞条件。
    }  # 本地合同汇总结束。


def _inspect_official_anchor_layout_metadata(seq_id: str, raw_root: Path) -> dict[str, Any]:  # 只检查官方元数据是否存在且结构可读。
    """检查官方锚点元数据是否存在，并按同一标准评估。"""  # 这里只看官方元数据本身，不和本地合同混在一起。
    official_candidate, official_source, official_source_paths = _discover_official_anchor_layout_candidate(seq_id, raw_root)  # 先找官方候选项。
    if official_candidate is None or official_source is None:  # 没找到就返回不可用报告。
        return {  # 官方元数据缺失时直接返回不可用报告。
            'metadata_available': False,  # 没找到官方元数据。
            'source': None,  # 来源为空。
            'source_paths': [],  # 来源路径为空。
            'anchor_ids_available': False,  # 编号不可用。
            'anchor_positions_available': False,  # 位置不可用。
            'position_dim': None,  # 维度未知。
            'blockers': ['official_anchor_metadata_unavailable'],  # 记录官方元数据不可用。
        }  # 官方元数据缺失报告结束。

    _, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 按统一规则评估官方候选项。
        official_candidate,  # 官方布局候选项。
        source=official_source,  # 官方来源标识。
        source_paths=official_source_paths,  # 官方来源路径。
    )  # 官方锚点元数据评估完成。
    return {  # 汇总官方锚点元数据的判断结果。
        'metadata_available': bool(anchor_layout_metadata is not None),  # 是否有元数据。
        'source': official_source,  # 官方来源标识。
        'source_paths': list(official_source_paths),  # 官方来源路径。
        'anchor_ids_available': bool(  # 是否有可用的锚点编号。
            isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
            and isinstance(anchor_layout_metadata.get('anchor_ids'), list)  # 再确认锚点编号是列表。
            and anchor_layout_metadata.get('anchor_ids')  # 最后确认列表非空。
        ),  # 锚点编号可用性判断结束。
        'anchor_positions_available': bool(  # 是否有可用的锚点位置。
            isinstance(anchor_layout_metadata, dict)  # 先确认元数据是字典。
            and isinstance(anchor_layout_metadata.get('anchor_positions'), list)  # 再确认锚点位置是列表。
            and anchor_layout_metadata.get('anchor_positions')  # 最后确认列表非空。
        ),  # 锚点位置可用性判断结束。
        'position_dim': anchor_layout_report.get('anchor_layout_position_dim'),  # 位置维度。
        'blockers': list(anchor_layout_report.get('anchor_layout_blockers') or []),  # 阻塞条件。
    }  # 官方元数据汇总结束。


def audit_miluv_official_sample(seq_id: str, raw_root) -> dict[str, Any]:  # 给单个官方样例生成审计结论。
    """单独审计官方 MILUV 样例元数据，不等同于本地锚点合同。"""  # 这个审计函数只说明情况，不在这里自动切换来源。
    if not seq_id:  # 序列编号不能为空。
        raise ValueError('seq_id must be a non-empty string')  # 序列编号不能为空。

    raw_root_path = Path(raw_root)  # 标准化原始根目录。
    seq_dir = raw_root_path / seq_id  # 拼出序列目录。
    local_anchor_layout_contract = _inspect_local_anchor_layout_contract(seq_id, raw_root_path)  # 检查本地锚点合同。
    official_anchor_metadata = _inspect_official_anchor_layout_metadata(seq_id, raw_root_path)  # 检查官方锚点元数据。

    local_contract_still_needed = list(local_anchor_layout_contract['blockers'])  # 先把本地合同的阻塞条件拷贝出来。
    if not local_contract_still_needed and not local_anchor_layout_contract['ready']:  # 没有阻塞项但又没就绪时，补一个说明。
        local_contract_still_needed = ['local_anchor_layout_contract_unmet']  # 明确表示本地合同仍未满足。

    if local_anchor_layout_contract['ready']:  # 本地合同满足时，状态最简单。
        status = 'local_contract_satisfied'  # 本地合同满足时，状态最简单。
    elif official_anchor_metadata['metadata_available']:  # 官方元数据存在但本地合同还没满足。
        status = 'official_metadata_present_local_contract_missing'  # 官方元数据存在但本地合同还没满足。
    else:  # 两边都没有满足。
        status = 'official_metadata_missing_local_contract_missing'  # 两边都没有满足。

    return {  # 汇总官方样例审计结果。
        'dataset_name': 'miluv',  # 数据集名称。
        'seq_id': seq_id,  # 序列编号。
        'seq_dir': str(seq_dir),  # 序列目录。
        'seq_dir_exists': seq_dir.is_dir(),  # 序列目录是否存在。
        'official_anchor_metadata': official_anchor_metadata,  # 官方元数据报告。
        'local_anchor_layout_contract': local_anchor_layout_contract,  # 本地合同报告。
        'official_metadata_used_for_local_contract': False,  # 这个审计函数只说明情况，不在这里自动切换来源。
        'local_contract_still_needed': local_contract_still_needed,  # 本地合同还缺什么。
        'status': status,  # 当前状态文本。
    }  # 审计结果结束。


def _inspect_miluv_sequence_readiness(seq_id: str, raw_root: Path) -> dict[str, Any]:  # 汇总单条 MILUV 序列是否能进入端到端烟雾流程。
    """检查一条 MILUV 序列是否满足端到端训练烟雾要求。"""  # 单条序列报告是总报告的最小组成块。
    seq_dir = raw_root / seq_id  # 拼出序列目录。
    stream_report, missing_streams = _probe_required_streams(seq_dir)  # 检查必需流。
    anchor_layout = _inspect_local_anchor_layout_contract(seq_id, raw_root)  # 检查本地锚点布局合同。

    reasons: list[str] = []  # 收集不就绪原因。
    if missing_streams:  # 只要缺流就记原因。
        reasons.append('missing_required_streams')  # 记录缺少必需流。
    if not anchor_layout['ready']:  # 锚点布局没就绪也要记原因。
        reasons.append('missing_anchor_layout')  # 记录缺少锚点布局。

    ready = not reasons  # 没有原因就说明就绪。
    return {  # 汇总单条 MILUV 序列的就绪判断。
        'seq_id': seq_id,  # 序列编号。
        'seq_dir': str(seq_dir),  # 序列目录。
        'status': 'ready' if ready else 'not_ready',  # 状态文本。
        'ready_for_end_to_end_training_smoke': ready,  # 是否可用于端到端训练烟雾测试。
        'reasons': _dedupe_preserve_order(reasons),  # 去重后的原因列表。
        'missing_streams': missing_streams,  # 缺失的流。
        'required_streams': stream_report,  # 各流检查报告。
        'anchor_layout': anchor_layout,  # 锚点布局报告。
    }  # 单条序列就绪报告结束。


def inspect_miluv_raw_readiness(raw_root) -> dict[str, Any]:  # 汇总整个 MILUV 原始根目录是否足够进行烟雾验证。
    """检查 MILUV 原始根目录是否满足当前端到端训练烟雾要求。"""  # 总报告把每条序列的状态聚合起来，给 prepare/eval 提供门禁判断。
    raw_root_path = Path(raw_root)  # 标准化原始根目录。
    sequence_dirs = _list_miluv_sequence_dirs(raw_root_path)  # 列出序列目录。
    sequence_reports = {  # 按序列编号收集每条序列报告。
        seq_dir.name: _inspect_miluv_sequence_readiness(seq_dir.name, raw_root_path)  # 生成单序列报告。
        for seq_dir in sequence_dirs  # 遍历所有候选序列目录。
    }  # 序列报告字典闭合。

    reasons: list[str] = []  # 收集整体不就绪原因。
    if not sequence_dirs:  # 没有任何序列目录时先记原因。
        reasons.append('missing_raw_sequence')  # 记录缺少原始序列。
    for seq_report in sequence_reports.values():  # 扫每条序列报告。
        if seq_report['status'] != 'ready':  # 不是 ready 的就把原因合进来。
            reasons.extend(seq_report['reasons'])  # 累加原因。

    ready_sequence_count = sum(  # 统计 ready 序列数量。
        1  # 每条 ready 序列记 1。
        for seq_report in sequence_reports.values()  # 遍历所有序列报告。
        if seq_report['ready_for_end_to_end_training_smoke']  # 只数可就绪的。
    )  # ready 统计结束。
    overall_ready = bool(sequence_dirs) and ready_sequence_count == len(sequence_reports)  # 有序列且全部就绪才算整体就绪。
    report = {  # 组装总报告。
        'dataset_name': 'miluv',  # 数据集名称。
        'raw_root': str(raw_root_path),  # 原始根目录。
        'raw_root_exists': raw_root_path.is_dir(),  # 根目录是否存在。
        'status': 'ready' if overall_ready else 'not_ready',  # 总状态。
        'gate_action': 'pass' if overall_ready else 'skipped',  # 门禁动作。
        'ready_for_end_to_end_training_smoke': overall_ready,  # 是否可进入端到端烟雾流程。
        'reasons': _dedupe_preserve_order(reasons),  # 去重后的原因。
        'sequence_count': len(sequence_reports),  # 序列数量。
        'ready_sequence_count': ready_sequence_count,  # 就绪序列数量。
        'required_streams': list(_REQUIRED_STREAM_FILES.keys()),  # 必需流名列表。
        'sequence_ids': list(sequence_reports.keys()),  # 序列编号列表。
        'sequences': sequence_reports,  # 每条序列的详细报告。
    }  # 总报告闭合。
    if not raw_root_path.is_dir():  # 根目录不是目录时补错误信息。
        report['raw_root_error'] = f'raw_root_not_found: {raw_root_path}'  # 记录错误字段。
    return report  # 返回总报告。


def read_miluv_sequence(
    seq_id: str,
    raw_root,
    field_mapping: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """读取一条 MILUV 序列，返回原始包和读取报告。

    参数：
        seq_id: 序列编号。
        raw_root: 原始数据根目录。
        field_mapping: 可选的字段映射定义，用于验证完整性。

    返回：
        Tuple[Dict, Dict]: 原始包和读取报告的二元组。

    异常：
        ValueError: seq_id 为空或 field_mapping 为空字典时抛出。
        FileNotFoundError: 序列目录不存在时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"seq_id": seq_id, "raw_root": str(raw_root), "has_field_mapping": field_mapping is not None}, "read_miluv_sequence 入口参数")
    if not seq_id:
        raise ValueError('seq_id must be a non-empty string')
    if field_mapping is not None and (not isinstance(field_mapping, dict) or not field_mapping):
        raise ValueError('field_mapping must be a non-empty mapping')

    seq_dir = Path(raw_root) / seq_id
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"MILUV sequence directory not found: {seq_dir}")

    bundle = {
        'imu_raw': _read_json_records(seq_dir / 'imu.json'),
        'uwb_raw': _read_json_records(seq_dir / 'uwb.json'),
        'vio_raw': _read_json_records(seq_dir / 'vio.json'),
        'gt_raw': _read_json_records(seq_dir / 'gt.json'),
    }
    anchor_layout = None  # 先不假设本地布局可用。
    anchor_layout_metadata = None  # 先不假设有可用元数据。
    anchor_layout_report = {  # 先准备一个默认布局报告，结构与 _assess_anchor_layout_candidate 返回值对齐。
        'anchor_layout_metadata_available': False,  # 先默认没有拿到元数据。
        'anchor_layout_metadata_source': None,  # 来源标识先空着。
        'anchor_layout_source_paths': [],  # 来源路径先空着。
        'anchor_layout_position_dim': None,  # 位置维度先未知。
        'anchor_layout_teacher_ready': False,  # 先默认不能给教师投影使用。
        'anchor_layout_blockers': [],  # 阻塞条件先空着。
    }  # 默认布局报告闭合。

    anchor_layout_candidate = _read_optional_json_object(seq_dir / 'anchor_layout.json')  # 先读本地布局文件；损坏时自动返回 None。
    if anchor_layout_candidate is not None:  # 本地布局存在时优先评估本地候选。
        anchor_layout, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 评估本地布局候选项。
            anchor_layout_candidate,  # 本地布局候选项。
            source=str(anchor_layout_candidate.get('source') or 'per_sequence_anchor_layout_json'),  # 来源字段。
            source_paths=[str(seq_dir / 'anchor_layout.json')],  # 本地布局文件路径。
        )  # 本地布局候选评估完成。
    else:  # 本地文件不存在时，转而看官方元数据。
        official_candidate, official_source, official_source_paths = _discover_official_anchor_layout_candidate(  # 否则尝试官方元数据。
            seq_id,  # 序列编号。
            Path(raw_root),  # 原始根目录。
        )  # 官方布局候选发现完成。
        if official_candidate is not None and official_source is not None:  # 找到官方候选才继续。
            anchor_layout, anchor_layout_metadata, anchor_layout_report = _assess_anchor_layout_candidate(  # 评估官方候选项。
                official_candidate,  # 官方布局候选项。
                source=official_source,  # 官方来源标识。
                source_paths=official_source_paths,  # 官方来源路径。
            )  # 官方布局候选评估完成。

    if anchor_layout is not None:  # 如果布局合同可用，就放进原始包。
        bundle['anchor_layout_raw'] = anchor_layout  # 把布局原始对象放进返回包。
    if anchor_layout_metadata is not None:  # 如果有更完整的元数据，也一起放进去。
        bundle['anchor_layout_metadata_raw'] = anchor_layout_metadata  # 把布局元数据也放进去。
    read_report = {  # 读取报告只做摘要。
        'dataset_name': 'miluv',  # 数据集名称。
        'seq_id': seq_id,  # 序列编号。
        'seq_dir': str(seq_dir),  # 序列目录。
        'streams': {key: len(value) for key, value in bundle.items() if key.endswith('_raw') and isinstance(value, list)},  # 各流行数。
        'is_complete': all(bundle[key] for key in ('imu_raw', 'uwb_raw', 'vio_raw', 'gt_raw')),  # 是否四个主流都读到了内容。
        'anchor_layout_available': anchor_layout is not None,  # 布局是否可用。
        **anchor_layout_report,  # 展开布局报告字段。
    }  # 读取报告闭合。
    return bundle, read_report  # 返回原始包和报告。
