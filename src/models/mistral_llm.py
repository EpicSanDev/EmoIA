import os
import aiohttp
import json
from typing import Dict, Any
from .local_llm import LocalLanguageModel  # On hérite de l'interface commune

class MistralLLM(LocalLanguageModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = getattr(config, "ollama_base_url", "http://localhost:11434")
        self.model_name = getattr(config, "mistral_model", "mistral")
        self.temperature = getattr(config, "temperature", 0.7)
        self.max_tokens = getattr(config, "max_tokens", 1024)

    async def generate_response(self, user_input: str, context: str = "", emotional_state: str = None, personality: str = None, language: str = "en") -> str:
        """
        Génère une réponse en utilisant le modèle Mistral via Ollama (version asynchrone)
        avec support multilingue.
        """
        # Ajouter une instruction de langue au prompt
        language_instruction = f"Respond in {language} language."
        full_prompt = f"{context}\n\n{language_instruction}\n\n{user_input}" if context else f"{language_instruction}\n\n{user_input}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    }
                ) as response:
                    # Vérifier le statut de la réponse
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Erreur Ollama {response.status}: {error_text}")
                    
                    full_response = ""
                    async for line in response.content:
                        if line:
                            chunk = json.loads(line)
                            if not chunk.get("done"):
                                full_response += chunk.get("response", "")
                    
                    return full_response.strip()
        
        except aiohttp.ClientError as e:
            return f"Erreur réseau: {str(e)}"
        except Exception as e:
            return f"Erreur lors de l'appel à Ollama: {str(e)}"
