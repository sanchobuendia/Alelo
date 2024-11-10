# Alelo

## Desafio de Aprendizado de Máquina

### Introdução
Esta é uma pequena série de exercícios para avaliar seu conhecimento em aprendizado de máquina. Responda às perguntas detalhando as etapas executadas para resolver cada tarefa. Todas as perguntas são simples, mas esta é sua chance de nos mostrar seus conhecimentos técnicos.

Esperamos receber seu código com sua solução e análise, incluindo quaisquer etapas de pré-processamento. Sinta-se à vontade para usar a linguagem de programação com a qual se sentir mais confortável, mas forneça instruções para executar seu código. Não é necessário codificar os algoritmos do zero – você pode usar qualquer biblioteca disponível, como o `scikit-learn`. Note que a organização e legibilidade do seu código também serão avaliadas.

Queremos que este desafio (e seu trabalho conosco) seja divertido, então selecionamos um conjunto de dados de super-heróis para você explorar! Você pode baixar os dados no [Kaggle](https://www.kaggle.com/claudiodavi/superhero-set/data). O conjunto é composto por dois arquivos CSV: `super_hero_powers.csv` e `heroes_information.csv`. Você usará ambos.

Este desafio foi projetado para durar no máximo três dias. Não estamos cronometrando, mas sugerimos que você não ultrapasse esse período. O objetivo não é obter o modelo de melhor desempenho, mas sim avaliar sua abordagem. Invista seu tempo explicando suas soluções, em vez de buscar o melhor modelo. A divisão de dados e as métricas de avaliação ficam a seu critério e serão avaliadas.

Caso tenha dúvidas, não hesite em nos contatar. No entanto, não comentaremos as suas soluções nem aprofundaremos nos detalhes técnicos das questões.

---

### Pré-Processamento

#### Dataset heroes_information

O dataset heroes_information possui 734 linhas, sendo 1 duplicada. Esta linha duplicado foi retirada.

Algumas features categóricas possuem uma string "-" indicando que o herói não possui a informação, valor nulo. Neste caso substituí o "-" pela string "Unknown". No entanto, em alguns casos as features: Eye color, Race e Hair color. São "Unknown", nos casos em que as 3 colunas não possuem informação eu retirei a linha do dataset, foram retiradas 5 linhas.

Para as variáveis numéricas (Height e Weight) existem muitos casos onde os valores são -99. Essas colunas possuem 29% e 32%, respectivamente, de casos onde o valor é -99. Optei por retirar do dataframe as linhas onde as duas colunas possuem o valor -99. Após isso, os que restaram foram substituídos pela média. Sei que a média pode camuflar algumas informações, mas decidi assumir o risco.

As análises de correlação tanto das variáveis numéricas, usando o pearson, quanto das categóricas, usando o Cramér's V. Não mostram nenhuma correlação que justificasse a exclusão de alguma variável.

Como pode ser visto nos plots, boxplot e scatterplot, as colunas: Height e Weight. Possuem outliers. Retirei os outliers usando o método do intervalo interquartil (IQR).

A coluna Skin color possui 90% de valores faltantes. Sendo assim, decidi remover esta coluna.

A coluna Race tem 35% de humanos, 35% valores faltantes e o restante distribuído entre outras raças. 

| Raça     | Proporção   |
|----------|-------------|
| Unknown  | 0.356061    |
| Human    | 0.356061    |
| Mutant   | 0.111111    |
| Android  | 0.012626    |
| Alien    | 0.012626    |

Temos duas raças com mais de 10% da base: Human e Mutant. Essa coluna não carrega informações relevantes, sendo assim, retirei elas da base de modelagem.

Alguns heróis possuem mesmo nome mas informações diferentes

| Name | Gender | Eye color | Race           | Hair color | Height | Publisher      | Alignment | Weight |
|------|--------|-----------|----------------|------------|--------|----------------|-----------|--------|
| Nova | Male   | brown     | Human          | Brown      | 185.0  | Marvel Comics  | good      | 86.0   |
| Nova | Female | white     | Human / Cosmic | Red        | 163.0  | Marvel Comics  | good      | 59.0   |

Não retirei da base esses casos.

Por fim a base ficou com 396 linhas.

#### Dataset Super Hero Powers

Quanto a base de poderes, não possui nulos nem as features são correlacionadas o suficiente para descartar variáveis. Desta forma não fiz alterações na base.

#### Arquivo python com o pré-processamento

Por fim, criei o arquivo preprocessing.py com as regras de pré-processamento que defini aqui.

### Clustering

**Questão 1**: Agrupe os super-heróis de acordo com seus poderes e informações. Execute um método de cluster não supervisionado, escolhendo o número de clusters que considerar mais adequado.

1. Qual algoritmo você escolheu e por quê?
2. Quais recursos você usou e por quê? Explique qualquer pré-processamento ou engenharia de recursos (seleção) que você executou.

**Resposta**: Como criei um arquivo .py para o pré-processamento eu fiz o merge entre as duas bases e apliquei o pré-processamento na base resultante.

Como são apenas 2 variáveis contínuas, numéricas, e o restante é categórica. O dataset de modelagem ficou com uma dimensionalidade muito grande. Para diminuir o impacto da dimensionalidade eu apliquei um PCA.

Peguei as variáveis com um corte de 80% na importâcia cumulativa.

Para escolher o número ótimo de clusters eu testei 3 diferentes aboradagens: teste do elbow, método da silhueta e o BIC.

* O Elbow Test indicou uma estabilização na redução da inércia a partir de 8 a 10 clusters, mas o BIC sugere penalização significativa a partir de 6 clusters, indicando que a complexidade pode ser excessiva com muitos clusters.
* O coeficiente de silhueta aponta que a separação ideal está com poucos clusters, sendo que o valor máximo é para 2 clusters, mas cai rapidamente em seguida, sugerindo que mais clusters podem comprometer a qualidade da separação.
* O BIC também sugere que entre 4 e 5 clusters você pode alcançar um bom equilíbrio entre explicabilidade e simplicidade.

Portanto, 4 a 5 clusters é a melhor quantidade recomendada, considerando a estrutura e as características dos seus dados após a aplicação do PCA.

**Questão 2**: Um dos desafios do clustering é definir o número certo de clusters. Como você escolheu esse número? Como você avalia a qualidade dos clusters finais?

**Resposta**: Após o PCA, as viráveis que ficaram são todas categóricas. Sendo assim, escolhi o K-Modes como modelo.

O K-Modes é um algoritmo de clusterização projetado especificamente para lidar com dados categóricos. É uma adaptação do algoritmo K-Means, que é voltado para dados numéricos, e utiliza medidas de similaridade adequadas a dados categóricos para agrupar elementos com características semelhantes.

---

### Identificando os Bandidos

Nesta seção, vamos trabalhar com um problema de aprendizagem supervisionada, formulando uma tarefa de classificação para prever o alinhamento dos super-heróis (bom ou mau).

**Questão 3**: Primeiramente, use o algoritmo Naive Bayes. Execute-o nos dados dos super-heróis para prever a variável de alinhamento e avalie os resultados. Detalhe qualquer pré-processamento e engenharia de recursos que você aplicou no processo.

- Quais hipóteses assumimos ao usar o algoritmo Naive Bayes?
- Como as características específicas deste conjunto de dados influenciam suas escolhas e resultados de modelagem?
- Como você avalia os resultados?

**Resposta**: Ao assumir o uso do Naive Bayes assumimos que as variáveis são independentes. O que foi verificado, não encontrei alta correlação entre as variáveis.
A alta dimensionalide pode ser um problemas para NB. O desbalanceamento da base pode prejudicar o desempenho. Se desconsiderarmos as variáveis continuas, que são apenas 2, poderíamos utilizar uma variaçao do NB, Bernoulli Naive Bayes.
O dataset possui algumas características que infleunciam na modelagems: alta dimensionalidade, apenas duas variáveis contínuas, poucas linhas, e uma distruibuiçao do target que é de 71-29. 

Devida a alta dimensionalidade eu calculei a feature importance e utilizei apenas as colunas mais importantes.

Os resultados não foram bons e o modelo "chuta" raticamente a mesma classe.

**Questão 4**: Agora, experimente o algoritmo de classificação que considerar mais adequado para essa tarefa.

1. O que motivou sua escolha do algoritmo?
2. Como esse algoritmo se compara ao Naive Bayes em relação às suposições e resultados da modelagem?

**Resposta**: A escolha de LightGBM é justificada pelo fato de ele ser mais robusto para lidar com datasets que possuem muitas variáveis categóricas. Além disso, LightGBM é uma escolha eficiente para conjuntos de dados pequenos e desbalanceados quando configurado com class_weight="balanced". Isso ajusta o modelo para que ele leve em conta a desproporção nas classes, tornando-o mais sensível à classe minoritária. LightGBM também é otimizado para manipular variáveis categóricas sem a necessidade de conversões complexas, o que melhora sua capacidade de capturar relações não lineares e características de alta cardinalidade.

Os resultados não foram bons o suficiente para colocar em produção. Mas representou uma bo melhora em relaçao ao Naive Bayes. Porque separou melhor as classes.

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
