"""Repeat all native fixtures on the independent worker before causal generation."""
from engine import OUT,model_and_tokenizer,write
import runtime_validation as rv
model,tok=model_and_tokenizer();rv.OUT=OUT/'worker_preflight';rv.OUT.mkdir(exist_ok=True)
for arm in ['base','clean','vireo']:
    rv.set_checkpoint(model,arm);rv.validate(model,tok,arm)
write(rv.OUT/'complete.json',{'native_cases':36,'all_passed':True})
