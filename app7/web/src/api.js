export async function generateAnswer(query) {
  const response = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  return response.json();
}

export async function generateDatasets() {
  const response = await fetch('/generate_datasets', { method: 'POST' });
  return response.json();
}

export async function runBasicFinetune() {
  const response = await fetch('/basic_finetune', { method: 'POST' });
  return response.json();
}

export async function runPeftFinetune() {
  const response = await fetch('/peft_finetune', { method: 'POST' });
  return response.json();
}

export async function runRlhfTrain() {
  const response = await fetch('/rlhf_train', { method: 'POST' });
  return response.json();
}
