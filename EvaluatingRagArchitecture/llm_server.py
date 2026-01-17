
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import uvicorn

app = FastAPI()
MODEL_NAME = "microsoft/phi-2"
print(f"Loading model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model: {MODEL_NAME} on {device}...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto", 
    trust_remote_code=True
)
pipe = pipeline( "text-generation", model=model,tokenizer=tokenizer,)
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
@app.post("/generate")
async def generate(request: GenerationRequest):
    outputs = pipe(
        request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    return {"text": outputs[0]['generated_text'],"model": MODEL_NAME,"device": str(model.device)}
@app.get("/health")
async def health():
    return {"status": "healthy","model": MODEL_NAME,"device": str(model.device),"cuda": torch.cuda.is_available(),
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)