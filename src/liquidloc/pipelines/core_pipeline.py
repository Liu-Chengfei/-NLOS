_NEURAL_METHODS = {MODEL_NAME_LSTM, MODEL_NAME_LIQUID, MODEL_NAME_TRANSFORMER}  # 需要模型参与的神经方法。
_CLASSICAL_METHODS = {ESTIMATOR_NAME_EKF}  # 纯估计器或经典方法。
_SCENE_AXES = SCENE_AXES  # 场景协议里六个主轴的固定顺序，来源为冻结协议。
_PARAM_ATTR_CANDIDATES = ('params', 'param_count', 'parameter_count', 'num_params')  # 参数数量字段的候选名字。
_RAM_ATTR_CANDIDATES = ('ram_peak', 'ram_peak_mb', 'peak_ram_mb', 'memory_peak_mb')  # 峰值内存字段的候选名字。
_MODEL_CFG_ROOT = find_project_root() / 'configs' / 'models'  # 模型配置目录根路径。