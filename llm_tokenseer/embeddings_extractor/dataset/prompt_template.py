from typing import Optional

from jinja2 import Template

PAD_TOKEN = "<|finetune_right_pad_id|>"
PAD_TOKEN_ID = 128004

template_str = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ question }}{{ context }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

template = Template(template_str)


def render_prompt(question: str, context: Optional[str] = None) -> str:
    context = context or ""
    context = f"\n{context}"
    return template.render(question=question, context=context)


print(render_prompt("Choose the best option", "Option 1, Option 2, Option 3"))
print(render_prompt("What is the capital of France?"))
