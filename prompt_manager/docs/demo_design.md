## 采用 *.prompt.yaml + *.prompt.md 双文件方案，IDE 零成本适配，生态工具完美复用。
## 使用 basename 隐式关联
## Markdown 只包含提示词正文和占位符，YAML 包含所有结构化配置。
## 设计并开发一个轻量级的 Prompt Manager，负责加载、校验、渲染、解析返回结果。
## yaml 协议设计：
1. param 字段定义参数，采用 python type hint 分隔，示例：
```
param:
    username:
        _type: str
        _desc: 用户名称
    age:
        _type: int
        _desc: 用户年龄
    tone:
        _type: str
        _default: 轻松愉快
        _desc: 输出语气偏好
    recent4chat:
        _type: list[dict]
        _default: []
        _desc: 最近 4 条对话记录
        _item_all:
            role: 
                _type: literal["assistant", "system", "user"]
                _desc: 发言的角色
            content:
                _type: str
                _desc: 发言的内容
    memory_point:
        _type: dict[str, list | str]
        _desc: 总结提炼的对用户的关键记忆点
        user_hobby:
            _type: list[str]
            _desc: 用户的爱好
        user_tone:
            _type: str
            _desc: 用户常用的表达语气
        user_mood:
            _type: str
            _desc: 用户当且心情
```
其中，用 _ 前缀来显示区分关键字和用户配置字段。核心的关键字有 _type 采用 type hint 表示类型（仅允许 str, int, float, list, dict 5 个基础类型，可以使用 | 语法（联合类型）和 Any(代表该参数不检查类型)，list 和 dict 的 [ ] 语法如果不写缺省，那么默认等价于 list[Any] 和 dict[str, Any]，处于 [ ] 语法内的 dict 和 list 不必嵌套写 [ ]，此时的检查与）、_desc （可选）配置字段解释、_default（可选）配置字段默认值（配置了该字段默认该参数是可选）
_item_* （针对 list[dict] 的特殊场景）配置字典列表内字典的键值约束，格式与 param 约束一致（如果要进行个别元素的约束，那么用 _item_0, _item_1, ... 配置对应索引位置的元素的类型约束，但这种配置方式下，索引必须连续且从 0 开始（不允许出现 _item_0, _item_2 这种间断跳跃），最后要用 _item_remain 配置其余的，否则则默认只允许数组的长度为能容纳最大的那个 item_* 的索引，或者使用例子中的 _item_all 配置全部，这种情况下，不允许在使用其他的 _item_* 字段），dict 类型的配置方式类似于 param 语法的嵌套格式

2. output 字段定义返回约束，可配置 mode (支持 text 和 json)，当 mode为 json 时，还需要配置 schema 字段，schema 字段的格式与 param 一致，如：
```
output:
    mode: text # 配置输出为纯文本
---
output:
    mode: json # 配置输出为 json
    schema: # 此时需要配置 json 格式
        content:
            _type: str
        confidence:
            _type: float
            _desc: 对回答内容的置信度
        reference:
            _type: list[str]
            _desc: 引用的列表
```

3. meta 字段定义元数据，支持 author, title, desc, version, tags, 一个例子如下：
```
author: steve
title: simple_demo
desc: 一个示例
versions: 1.0.0
tags:
    - test
    - demo
``` 
## 一些实现的细节要求
1. Pydantic 深度集成：将参数定义直接映射为 Pydantic BaseModel 字段，使得参数校验和 JSON 输出校验完全依赖 pydantic；甚至可动态生成参数模型类。
2. YAML 解析请使用 yaml.safe_load，禁止执行任意指令；若不需要高级 YAML 功能，可考虑 StrictYAML 等简化解析库来避免隐式类型陷阱。
3. 占位符解析时需精确匹配 ${param} 格式；建议使用正则匹配并检查完整一致性。防止参数名冲突（如一个参数名为 id，另一个为 userid）时出现错误替换。
4. JSON Schema 嵌入 YAML 时，避免在 Schema 中使用 YAML 特殊语法（如别名、复杂标签）
5. 自动生成输出约束 Prompt
```yaml
output:
  mode: json

  schema:
    summary: str
    score: int
```
运行时自动拼接：
"""
## 输出约束
返回结果严格遵循 JSON 格式：
```Schema:
{
  "summary": "string",
  "score": "integer"
}
```
"""

## prompt 文件加载关系设计
读取 yaml
↓
读取 markdown
↓
解析 placeholder
↓
校验参数
↓
编译 PromptDefinition
↓
缓存（考虑对 yaml / md 文件 hash 来决定是否需要重新加载还是直接读取缓存文件，降低 demo 阶段多次启停调试的时间成本）