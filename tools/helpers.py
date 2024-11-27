from aleph_alpha_client import Client, CompletionRequest, Prompt

def make_llm_request(prompt: str, client: Client, MODEL: str, temperature: float=0.3) -> str:
    request = CompletionRequest(
        prompt=Prompt.from_text(prompt),
        temperature=temperature,
    )

    # API reference for the client:
    # https://aleph-alpha-client.readthedocs.io/en/latest/
    response = client.complete(request, model=MODEL)
    return response.completions[0].completion
