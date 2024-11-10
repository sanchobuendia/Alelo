# Alelo

## Desafio de Aprendizado de Máquina

### Introdução
Esta é uma pequena série de exercícios para avaliar seu conhecimento em aprendizado de máquina. Responda às perguntas detalhando as etapas executadas para resolver cada tarefa. Todas as perguntas são simples, mas esta é sua chance de nos mostrar seus conhecimentos técnicos.

Esperamos receber seu código com sua solução e análise, incluindo quaisquer etapas de pré-processamento. Sinta-se à vontade para usar a linguagem de programação com a qual se sentir mais confortável, mas forneça instruções para executar seu código. Não é necessário codificar os algoritmos do zero – você pode usar qualquer biblioteca disponível, como o `scikit-learn`. Note que a organização e legibilidade do seu código também serão avaliadas.

Queremos que este desafio (e seu trabalho conosco) seja divertido, então selecionamos um conjunto de dados de super-heróis para você explorar! Você pode baixar os dados no [Kaggle](https://www.kaggle.com/claudiodavi/superhero-set/data). O conjunto é composto por dois arquivos CSV: `super_hero_powers.csv` e `heroes_information.csv`. Você usará ambos.

Este desafio foi projetado para durar no máximo três dias. Não estamos cronometrando, mas sugerimos que você não ultrapasse esse período. O objetivo não é obter o modelo de melhor desempenho, mas sim avaliar sua abordagem. Invista seu tempo explicando suas soluções, em vez de buscar o melhor modelo. A divisão de dados e as métricas de avaliação ficam a seu critério e serão avaliadas.

Caso tenha dúvidas, não hesite em nos contatar. No entanto, não comentaremos as suas soluções nem aprofundaremos nos detalhes técnicos das questões.

---

### Clustering

**Questão 1**: Agrupe os super-heróis de acordo com seus poderes e informações. Execute um método de cluster não supervisionado, escolhendo o número de clusters que considerar mais adequado.

1. Qual algoritmo você escolheu e por quê?
2. Quais recursos você usou e por quê? Explique qualquer pré-processamento ou engenharia de recursos (seleção) que você executou.

**Questão 2**: Um dos desafios do clustering é definir o número certo de clusters. Como você escolheu esse número? Como você avalia a qualidade dos clusters finais?

---

### Identificando os Bandidos

Nesta seção, vamos trabalhar com um problema de aprendizagem supervisionada, formulando uma tarefa de classificação para prever o alinhamento dos super-heróis (bom ou mau).

**Questão 3**: Primeiramente, use o algoritmo Naive Bayes. Execute-o nos dados dos super-heróis para prever a variável de alinhamento e avalie os resultados. Detalhe qualquer pré-processamento e engenharia de recursos que você aplicou no processo.

- Quais hipóteses assumimos ao usar o algoritmo Naive Bayes?
- Como as características específicas deste conjunto de dados influenciam suas escolhas e resultados de modelagem?
- Como você avalia os resultados?

**Questão 4**: Agora, experimente o algoritmo de classificação que considerar mais adequado para essa tarefa.

1. O que motivou sua escolha do algoritmo?     
2. Como esse algoritmo se compara ao Naive Bayes em relação às suposições e resultados da modelagem?

---

### Além do Bem e do Mal

**Questão 5**: Transforme o problema em uma tarefa de regressão e tente prever o peso dos super-heróis com base nos outros recursos.

1. Qual algoritmo você escolheu e por quê?
2. Como você avalia o desempenho do seu algoritmo neste caso?

---

### Análise

**Questão 6**: Quais aspectos desse conjunto de dados apresentam problemas para agrupamento, classificação e regressão? Como você resolveu esses problemas?

---

### Documentação e Instruções

Forneça comentários claros no código e inclua uma seção explicando como usar as diferentes funcionalidades.
