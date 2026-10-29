# Previsão de doença cardíaca a partir de exames simples e hábitos de vida

Projeto Final de Machine Learning. Construímos um modelo de **classificação binária** que estima se um paciente tem doença cardíaca a partir de dados que ele mesmo consegue informar (idade, sexo, pressão, colesterol, tipo de dor no peito etc.), e o disponibilizamos por uma **API** consumida por uma **interface** (formulário).

> **Em uma frase:** dado o perfil clínico, a idade, o colesterol e a pressão arterial de um paciente, prever a ocorrência de doença cardíaca **para que** a equipe médica possa priorizar exames preventivos.

## Membros do grupo

| Nome 
| --- 
| Ana Clara Miguel
| Vinicius Henrique


---

## 1. Finalidade do projeto

Exames especializados do coração têm custo e tempo de espera, então saber quem tem maior chance de doença ajuda a decidir quem deve fazê-los primeiro. A ideia é criar uma **ferramenta de apoio à triagem**: a pessoa (ou o profissional) preenche um formulário curto e recebe uma estimativa do risco, que ajuda a decidir **quem deve ser encaminhado primeiro para exames preventivos**.

**Quem usaria:** profissionais de saúde em uma triagem inicial, ou qualquer pessoa que queira ter uma noção de risco antes de procurar atendimento.

**O que o projeto NÃO é:** não é diagnóstico e não substitui avaliação médica. Ele apenas ajuda a **priorizar**.

### Como o projeto funciona

```
┌────────────┐   JSON    ┌───────────────────┐   probabilidade   ┌─────────────┐
│ Formulário │ ────────► │ API (FastAPI)     │ ────────────────► │  Resposta   │
│ (interface)│           │ valida os campos  │                   │ ao usuário  │
└────────────┘           │ e chama o modelo  │                   └─────────────┘
                         └───────────────────┘
                            models/pipeline.joblib
```

A interface **consulta o modelo pela API, via HTTP**. Ela nunca carrega o arquivo do modelo diretamente.

**O que entra** (somente o que o usuário consegue informar):

| Campo | Tipo | Valores |
| --- | --- | --- |
| `age` | inteiro | idade em anos (faixa a fechar) |
| `sex` | texto | `M` ou `F` |
| `chest_pain_type` | texto | `TA`, `ATA`, `NAP` ou `ASY` |
| `resting_bp` | inteiro | pressão em repouso, mm Hg (faixa a fechar) |
| `cholesterol` | inteiro | colesterol, mg/dl (faixa a fechar) |
| `fasting_bs` | 0 ou 1 | glicemia em jejum acima de 120 mg/dl |
| `max_hr` | inteiro | frequência cardíaca máxima, bpm (faixa a fechar) |
| `exercise_angina` | texto | `Y` ou `N` |

**O que sai** (exemplo; o formato final será fechado com a API):

```json
{
  "modelo": "regressao_logistica",
  "versao": "0.1.0",
  "probabilidade_doenca": 0.82,
  "classificacao": "risco elevado",
  "mensagem": "Recomenda-se priorizar exames preventivos. Esta ferramenta não substitui avaliação médica."
}
```

---

## 2. Problema de Machine Learning

| Item | Definição |
| --- | --- |
| Tipo | Classificação binária |
| Alvo | `HeartDisease` (1 = tem doença cardíaca, 0 = não tem) |
| Classe positiva | `1` (doença) |
| Entradas | 8 colunas que o usuário consegue informar (tabela acima) |
| Colunas descartadas | `RestingECG`, `Oldpeak` e `ST_Slope`, por dependerem de exame especializado |
| Métrica principal | **Recall da doença**, com piso de especificidade |
| Baselines | Dummy, regra de uma variável e regressão logística |
| Validação | Divisão 80/20 estratificada + validação cruzada de 5 partições no treino |

### Por que o recall é a métrica principal

Para a equipe médica, **deixar passar um paciente doente (falso negativo) é pior do que chamar um saudável para um exame a mais (falso positivo)**. O grupo decidiu aceitar mais alarmes falsos para perder menos doentes.

Só o recall engana: como 55% dos pacientes têm doença, um "modelo" que responde *sempre doente* tem recall 1,00 sem aprender nada. Por isso a métrica vem com um **piso de especificidade** (valor inicial de 0,70, a ser ajustado pelo grupo):

| Métrica | O que mede |
| --- | --- |
| **Recall** (sensibilidade) | De cada 100 doentes, quantos o modelo encontra |
| **Especificidade** | De cada 100 saudáveis, quantos o modelo libera (precisa ficar acima do piso) |
| Precisão | De quem foi marcado como doente, quantos realmente são |
| F1 | Equilíbrio entre precisão e recall |
| ROC AUC | Qualidade do ranqueamento, independente do limiar |

A acurácia não é usada como critério, porque trata os dois tipos de erro como se custassem o mesmo.

---

## 3. Dados

- **Dataset:** Heart Failure Prediction Dataset (Kaggle).
- **Link da fonte:** (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download)
- **Tamanho:** 918 linhas × 12 colunas, sem duplicatas e sem valores vazios declarados.
- **Balanceamento:** 508 pacientes com doença (55%) e 410 sem (45%).
- **Perfil:** 725 homens e 193 mulheres (cerca de 79% homens).

### Dicionário das colunas de entrada

| Coluna | Significado | Valores |
| --- | --- | --- |
| `Age` | Idade | anos |
| `Sex` | Sexo | `M` ou `F` |
| `ChestPainType` | Tipo de dor no peito | `TA` angina típica, `ATA` angina atípica, `NAP` dor não anginosa, `ASY` assintomático |
| `RestingBP` | Pressão arterial em repouso | mm Hg |
| `Cholesterol` | Colesterol sérico | mg/dl |
| `FastingBS` | Glicemia em jejum acima de 120 mg/dl | `1` sim, `0` não |
| `MaxHR` | Frequência cardíaca máxima atingida | bpm |
| `ExerciseAngina` | Dor no peito induzida por exercício | `Y` ou `N` |
| `HeartDisease` | **Alvo**: doença cardíaca | `1` ou `0` |

_Conferir este dicionário com a descrição da página do Kaggle._

### Tratamento dos dados

- **Zeros impossíveis:** 172 pacientes têm colesterol igual a `0` e 1 tem pressão igual a `0`. Como isso não existe em paciente vivo, tratamos como **dado ausente** (nossa interpretação) e preenchemos com a mediana, aprendida só no treino.
- **Teste trancado:** a divisão 80/20 (734 linhas de treino e 184 de teste) é feita logo depois de carregar os dados, com `stratify=y` e `random_state=42`. O conjunto de teste **só é aberto na Entrega 3**, uma única vez.
- **Sem vazamento:** o pré-processamento fica dentro de um `Pipeline` do scikit-learn, então mediana, escala e *encoding* são aprendidos apenas com os dados de treino de cada rodada da validação cruzada.

---

## 4. Resultados até agora (Entrega 1)

Baselines medidos por validação cruzada de 5 partições, **somente no treino** (média ± desvio padrão). O piso de especificidade é 0,70.

| Baseline | Recall | Especificidade | Precisão | F1 | ROC AUC | Atende o piso? |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Dummy (sempre "doença") | 1,00 ± 0,00 | 0,00 ± 0,00 | 0,55 ± 0,00 | 0,71 ± 0,00 | 0,50 ± 0,00 | não |
| 2. Regra de uma variável | 0,78 ± 0,03 | 0,74 ± 0,07 | 0,79 ± 0,05 | 0,78 ± 0,03 | 0,76 ± 0,04 | sim |
| 3. Regressão logística | **0,83 ± 0,05** | 0,78 ± 0,06 | 0,83 ± 0,04 | 0,83 ± 0,04 | 0,89 ± 0,03 | sim |

**Leitura:**

- O **Dummy** encontra todos os doentes só porque chama todo mundo de doente. Ele falha no piso e não serve.
- A **regra de uma variável** que a árvore aprendeu sozinha é: *dor no peito do tipo assintomático (`ASY`) → doença*. Já sozinha, essa variável separa bem os dois grupos.
- A **regressão logística** é o modelo a superar. No treino, ela deixa passar 70 doentes (falsos negativos) e gera 71 alarmes falsos.

> Os números vêm do notebook `00_primeiro_contato.ipynb` com `random_state=42`. Podem variar um pouco conforme a versão das bibliotecas.

---

## 5. O que vamos fazer

| Etapa | Atividades |
| --- | --- |
| **Entrega 1** (proposta) | Escolher o dataset, definir problema, alvo, métrica e baseline, criar o repositório. **Feito neste README e no notebook 00.** |
| **Análise exploratória** | Aprofundar nos dados de treino: relação de cada variável com a doença, outliers, correlações. Notebook `01_eda.ipynb`. |
| **Modelagem** | Testar modelos mais fortes que a regressão logística (árvore, Random Forest, gradient boosting), ajustar hiperparâmetros por validação cruzada e **escolher o limiar de decisão** que favorece o recall sem derrubar a especificidade. Notebook `02_modelo.ipynb`. |
| **Entrega 3** (avaliação final) | Abrir o conjunto de teste **uma única vez**, com o modelo e o limiar já escolhidos. Reportar métricas por sexo, matriz de confusão e limitações. |
| **API** | Salvar o pipeline completo com `joblib`; criar a API com FastAPI e validação de entrada com `pydantic`, devolvendo nome e versão do modelo. |
| **Interface** | Formulário em Streamlit (ou HTML/JS) que consulta a API via HTTP. |
| **Fechamento** | README com "como rodar a aplicação" em 3 comandos e commits de todos os integrantes. |

_Ajustar as datas e a numeração das entregas conforme o cronograma da disciplina._

### Decisões em aberto

- **Piso de especificidade e recall-alvo:** os valores iniciais (0,70 e 0,90) são do grupo e podem mudar. Nos testes, exigir recall 0,90 derrubou a especificidade para cerca de 0,62; será preciso escolher entre aceitar menos recall ou mais alarmes falsos.
- **`FastingBS`, `MaxHR` e `ExerciseAngina`:** também dependem de exame de sangue ou teste de esforço. Por ora ficam no modelo, mas vale rediscutir se o usuário realmente consegue informá-las.
- **Faixas válidas dos campos** da API (idade, pressão, colesterol e frequência cardíaca).

---

## 6. Estrutura do repositório

```
projeto-ml/
├── README.md                  ← proposta + como rodar
├── requirements.txt
├── data/                      ← CSV do dataset (ver "Como rodar")
├── notebooks/
│   ├── 00_primeiro_contato.ipynb   ← Entrega 1: dados, split, métrica, baselines
│   ├── 01_eda.ipynb
│   └── 02_modelo.ipynb
├── models/                    ← pipeline.joblib
├── api/
│   └── main.py
└── app/                       ← interface
```

---

## 7. Como rodar

> Esta seção cresce a cada entrega e termina com a aplicação em 3 comandos.

**Estado atual (notebooks)**

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloque o CSV do dataset na pasta `data/` (o nome do arquivo deve começar com `heart`, por exemplo `heart_1.csv`).
3. Abra e rode, em ordem, o notebook:
   ```bash
   jupyter notebook notebooks/00_primeiro_contato.ipynb
   ```

**No Google Colab**

1. Envie o `.ipynb` em *Arquivo → Fazer upload de notebook*.
2. Ao rodar a célula de carga, se o CSV não for encontrado, o notebook abre um botão para você selecioná-lo. Também é possível arrastar o CSV para `/content` pelo painel de arquivos.
3. O Colab apaga os arquivos ao reiniciar a sessão; para não reenviar, guarde o CSV no Google Drive e ajuste o caminho na célula de configuração.

**Entrega final (a preencher)**

```bash
# 1. instalar dependências
# 2. subir a API
# 3. abrir a interface
```

---

## 8. Boas práticas adotadas

| Prática | Como aplicamos | Situação |
| --- | --- | --- |
| Reprodutibilidade | `random_state=42` em todo o projeto e `requirements.txt` | em andamento |
| Baseline | Dummy, regra de uma variável e regressão logística, comparados antes de qualquer modelo complexo | feito |
| Métrica definida antes | Recall da doença com piso de especificidade, justificada na seção 2 | feito |
| Teste trancado | Divisão estratificada logo após carregar; teste só aberto na Entrega 3 | feito |
| Validação de entrada | Schema na API (tipos e faixas) | a fazer |
| Versão do modelo | Nome e versão na resposta da API | a fazer |
| Dados documentados | Fonte, licença e dicionário das colunas | falta registrar link e licença |
| Uso responsável | Colunas sensíveis revisadas e limites avisados (seção 9) | em andamento |

---

## 9. Uso responsável e limites

- **Não é diagnóstico.** A ferramenta apoia a priorização de exames e não substitui avaliação médica.
- **Base pequena e internacional.** São 918 pacientes, e o dataset não é brasileiro; os resultados **não** devem ser tratados como válidos para a população do Brasil.
- **Desequilíbrio entre sexos.** Cerca de 79% dos pacientes são homens, então o modelo pode errar mais em mulheres. Vamos reportar o desempenho **separado por sexo** na avaliação final.
- **Associação não é causa.** O modelo encontra padrões nos dados, não explica por que a doença acontece.
- **Falsos negativos existem.** Mesmo com foco em recall, o modelo deixa passar alguns doentes; um resultado de "baixo risco" não garante saúde.
- **Dados ausentes.** Os valores `0` de colesterol e pressão foram tratados como ausentes; essa é uma decisão nossa e fica registrada aqui.
- **Privacidade.** O dataset é público e anonimizado. Na aplicação, não vamos armazenar os dados digitados pelo usuário.

---

## 10. Referências

- Heart Failure Prediction Dataset, Kaggle: _colar o link da página_
- scikit-learn: https://scikit-learn.org
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://streamlit.io
