La génération se fait avec OpenAI dans les envs qui est déjà robuste aux images dans prompt : ok pour ça

Il faut juste check dans le trainer je pense

Il faut que dans l'input il y ait images en optionnel qui sera en input de la processing classe ?


Est-ce que dans _prepare_inputs il faut gérer les images ? ou seulement dans _get_per_token_logps ?