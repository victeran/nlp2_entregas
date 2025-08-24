# README – Trabajos Prácticos de Procesamiento de Lenguaje Natural 2

Este repositorio contiene tres trabajos prácticos que exploran diferentes técnicas y arquitecturas para modelos generativos, chatbots y sistema de agente.

📌 Trabajo 1 — Análisis de Estrategias de Decodificación con TinyGPT y MoE
Descripción

En este trabajo se estudió la generación de secuencias con un modelo base (TinyGPT) y una versión mejorada con arquitectura Mixture of Experts (MoE).

Objetivos

Evaluar cómo afectan diferentes técnicas de muestreo y búsqueda en la calidad del texto generado.

Comparar el rendimiento entre un modelo tradicional (TinyGPT) y una variante mejorada con MoE.

Identificar limitaciones en la generación de lenguaje con tokenización por caracteres.

Técnicas de Decodificación Analizadas

Greedy Search → Genera resultados predecibles, pero con riesgo de repeticiones.

Muestreo Aleatorio → Mayor diversidad, pero sacrifica coherencia.

Muestreo con Temperatura → Controla la variabilidad de las respuestas.

Top-K Sampling → Restringe la generación a los k tokens más probables.

Top-P (Nucleus) Sampling → Ajusta dinámicamente el rango de tokens considerados.

Resultados

Existe un trade-off entre diversidad y coherencia:

Mayor aleatoriedad ⇒ más creatividad, menos sentido.

Mayor determinismo ⇒ coherencia estable, riesgo de repeticiones.

La tokenización por caracteres limita fuertemente la calidad gramatical.

El modelo MoE logra mayor variabilidad en las predicciones, incluso usando métodos deterministas como greedy.

El uso de múltiples expertos sugiere que, con una mejor tokenización, se podrían obtener mejoras aún más significativas.


📌 Trabajo 2 — Implementación de Chatbot con RAG (Retrieval-Augmented Generation)
Descripción

En este trabajo se implementó un chatbot inteligente capaz de recuperar información de documentos y generar respuestas más completas utilizando la técnica de Retrieval-Augmented Generation (RAG).

La información de los CVs se preprocesa y se convierte en embeddings mediante Sentence-Transformers. Estos embeddings se almacenan en índices vectoriales de Pinecone, que permiten realizar búsquedas semánticas rápidas y precisas.

Cuando el usuario realiza una consulta, el chatbot:

Convierte la pregunta en un embedding.

Busca en Pinecone los fragmentos más relevantes del CV.

Usa ese contexto recuperado para generar una respuesta enriquecida y coherente.

Objetivos

Combinar búsqueda semántica con generación de texto para mejorar la precisión de las respuestas.

Utilizar Pinecone como vector store para almacenar y recuperar información de manera eficiente.

Desarrollar una interfaz interactiva con Streamlit para facilitar las consultas.



📌 Trabajo 3 — Chatbot con Nodo Decisor para Consulta de Múltiples CVs
Descripción

Este trabajo es una extensión del Trabajo 2, en la que se implementó un chatbot mejorado que puede responder consultas sobre diferentes CVs utilizando un único agente.

La clave de esta implementación es la incorporación de un nodo decisor, que analiza la consulta del usuario y determina qué índice de Pinecone debe consultar para recuperar el contexto adecuado. Una vez que se obtiene la información relevante, el mismo agente genera la respuesta final.

Funcionamiento

Análisis de la consulta → El nodo decisor detecta si en la pregunta se menciona un nombre específico.

Selección del índice →

Si se menciona a una persona → Se consulta el índice correspondiente en Pinecone.

Si no se menciona a nadie → Por defecto, se consulta el índice del alumno.

Recuperación de contexto → Se buscan los fragmentos más relevantes del CV desde el índice seleccionado.

Generación de la respuesta → El agente utiliza el contexto recuperado para construir una respuesta clara y precisa.

Objetivos

Extender la funcionalidad del chatbot para responder consultas sobre múltiples CVs.

Implementar un nodo decisor que seleccione el índice correcto en Pinecone según la consulta.

Mantener un único agente para simplificar la arquitectura, pero con acceso a múltiples fuentes de información.

