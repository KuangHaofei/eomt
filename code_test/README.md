# Code Test 文件夹

这个文件夹包含了EoMT项目中DINOv3集成过程中创建的所有测试文件、demo文件和分析文档。

## 📁 文件分类

### 🎮 DINOv3 Demo文件
- `dinov3_demo.py` - 完整的DINOv3演示，包含特征提取、相似性计算和注意力可视化
- `dinov3_quick_start.py` - DINOv3快速入门示例
- `dinov3_simple_demo.py` - 简化版DINOv3演示（无可视化）
- `dinov3_usage_examples.py` - DINOv3实用示例和封装类

### 🧪 测试文件
- `test_eomt_dinov3.py` - EoMT + DINOv3基础集成测试
- `test_resolution_support.py` - 多分辨率支持测试
- `test_feature_comparison.py` - 原版DINOv3 vs EoMT特征对比
- `test_layer_by_layer.py` - 逐层特征对比测试
- `test_layer_difference.py` - 精确定位特征差异起始层
- `test_same_input_comparison.py` - 相同输入条件下的对比测试
- `test_all_layers_comparison.py` - 完整24层对比测试
- `test_masked_attention_impact.py` - Masked Attention影响详细分析

### 📚 文档和总结
- `DINOv3_使用说明.md` - DINOv3使用指南
- `DINOv3_项目总结.md` - DINOv3 demo项目总结
- `EoMT_DINOv3_适配总结.md` - EoMT适配DINOv3的详细总结
- `EoMT_DINOv3_特征对比总结.md` - 特征对比测试总结
- `分辨率支持修复总结.md` - 分辨率支持修复过程总结
- `特征差异分析总结.md` - 特征差异的深度分析

## 🎯 主要成果

通过这些测试和分析，我们成功验证了：

1. ✅ **EoMT完美集成DINOv3** - 替换DINOv2为DINOv3
2. ✅ **多分辨率支持** - 支持224x224到640x640等任意分辨率
3. ✅ **特征一致性** - 前20层与原版DINOv3完全一致
4. ✅ **Masked Attention正常工作** - 正确实现query-backbone交互
5. ✅ **架构正确性** - 所有组件都按预期工作

## 🚀 使用说明

### 运行DINOv3 Demo
```bash
# 快速入门
python code_test/dinov3_quick_start.py

# 完整演示
python code_test/dinov3_demo.py

# 实用示例
python code_test/dinov3_usage_examples.py
```

### 运行测试
```bash
# 基础功能测试
python code_test/test_eomt_dinov3.py

# 分辨率支持测试
python code_test/test_resolution_support.py

# 特征对比测试
python code_test/test_feature_comparison.py

# Masked Attention分析
python code_test/test_masked_attention_impact.py
```

## 📝 注意事项

- 所有测试都需要在`eomt` conda环境中运行
- 确保DINOv3权重文件已正确下载到`models/dinov3/weights/`目录
- 某些测试可能需要较大的GPU内存

## 🎉 项目状态

**EoMT + DINOv3集成已完成！** 所有核心功能都已验证正常工作。
