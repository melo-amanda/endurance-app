import array
import streamlit as st
import pandas as pd
from datetime import datetime
from pulp import *
from typing import Tuple
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt

st.write("## Calculadora de Reposição de Nutrientes")

st.write("### Dados do Paciente")
pacCol1,pacCol2 = st.columns(2)
body_weight = pacCol1.number_input("Peso do Atleta (Kg)")
duration_activity = pacCol1.number_input("Duração prevista da atividade (h)")
percent_max_weight_loss = pacCol2.number_input("Perda máxima de peso (2% W, kg)")
sweat_rate = pacCol2.number_input("Taxa de Sudorese (L/h)")

st.write("### Limites de Reposição")
data = {"nutriente":["Teor de CHO (g/h)","Teor de Na+ (mg/L)","Teor de K+ (mg/L)","Teor de Cl- (mg/L)", "Teor de Ca+ (mg/L)", "Teor de Mg+ (mg/L)", "Teor de Fósforo (mg/L)", "Teor de Cafeína(mg/kg)"],
        'Unidade':["gramas por hora de atividade","mg por litro de água ingerida","mg por litro de água ingerida","mg por litro de água ingerida","mg por litro de água ingerida","mg por litro de água ingerida","mg por litro de água ingerida","mg por kg de peso corporal"],
        "limite_inferior_hard":[0,0,0,0,0,0,0,0],
        "limite_superior_hard":[0,0,0,0,0,0,0,0],
        "limite_inferior_soft":[0,0,0,0,0,0,0,0],
        "limite_superior_soft":[0,0,0,0,0,0,0,0],
        "peso":[0,0,0,0,0,0,0,0]}

caminho_arquivo_nutri = "/Users/amandamelo/Documents/Developer/enduranceApp/Input_Dados.xlsx"
df_limites = pd.read_excel(caminho_arquivo_nutri,sheet_name = "Limites", usecols="A:G", nrows=20)
df_limites_input = st.data_editor(df_limites)

st.write("### Suplementos Disponíveis")

df_supplements = pd.read_excel(caminho_arquivo_nutri,sheet_name = "composicaonutricional", usecols="A:P", nrows=20)
df_supplements_input = st.data_editor(df_supplements)

#Função que nos dá a suplementação ótima
def optimal_supplementation():

    #Input de informações do atleta e da atividade que ele irá necessitar de suplemento
    maximum_weight_loss = percent_max_weight_loss * body_weight #Conta de quantos quilos poderão ser perdidos na atividade
    total_sweat_loss = sweat_rate * duration_activity #Total de litros de suor que serão perdidos em toda a duração da atividade
    recommended_water_intake = total_sweat_loss - maximum_weight_loss #Dado que se pode perder um tanto de peso, quanto teria que ser ingerido de água considerando o tanto de suor perdido na atividade
    recommended_water_intake_rate = recommended_water_intake/duration_activity #Informação de ingestão de água recomendada por hora de atividade

    #Informações nutricionais dos suplementos
    df_supplements_fraction = df_supplements_input[df_supplements_input['Permite Fração?'] == 'Sim'] #Filtrar apenas os suplementos que permitem fracionamento
    df_supplements_integer = df_supplements_input[df_supplements_input['Permite Fração?'] == 'Não'] #Filtrar apenas os suplementos que NÃO permitem fracionamento
    matrix_composition_integer = df_supplements_integer.iloc[:, 4:12].values
    matrix_composition_fraction =  df_supplements_fraction.iloc[:, 4:12].values

    #Peso de cada nutriente
    peso = df_limites_input['peso'].tolist()

    #Tratamento dos limites de nutrientes: os limites são dados por mg/L.Se eu tenho que ingerir 3 litros de água, preciso deixar
    # o sangue com a mesma quantidade de eletrólitos por litro de líquido. Se eu ingiro água sem nada, fico com uma menor CONCENTRAÇÃO de eletrólitos no sangue
    df_limits_data = df_limites_input.copy()

    for idx, row in df_limits_data.iterrows():
        if "por hora de atividade" in row['Unidade']:
            for col in ['limite_inferior_hard', 'limite_superior_hard', 'limite_inferior_soft', 'limite_superior_soft']:
                if row[col] != "Sem limite":
                    df_limits_data.at[idx, col] = row[col] * duration_activity
        elif "por litro de água ingerida" in row['Unidade']:
            for col in ['limite_inferior_hard', 'limite_superior_hard', 'limite_inferior_soft', 'limite_superior_soft']:
                if row[col] != "Sem limite":
                    df_limits_data.at[idx, col] = row[col] * recommended_water_intake
        elif "por kg de peso corporal" in row['Unidade']:
            for col in ['limite_inferior_hard', 'limite_superior_hard', 'limite_inferior_soft', 'limite_superior_soft']:
                if row[col] != "Sem limite":
                    df_limits_data.at[idx, col] = row[col] * body_weight


    # Criando listas para armazenar os limites
    hard_lower_limit = []
    hard_upper_limit = []
    soft_lower_limit = []
    soft_upper_limit = []

    # Preenchendo as listas com as respectivas colunas do DataFrame
    for idx, row in df_limits_data.iterrows():
        hard_lower_limit.append(row['limite_inferior_hard'])
        hard_upper_limit.append(row['limite_superior_hard'])
        soft_lower_limit.append(row['limite_inferior_soft'])
        soft_upper_limit.append(row['limite_superior_soft'])

    #Iniciando a otimização
    modelo_det = LpProblem(name="Supplementation_Optmizer", sense=LpMinimize) #Cria o modelo

    x_len = len(matrix_composition_integer)
    y_len = len(matrix_composition_fraction)
    a = len(hard_upper_limit)
    P_X = list(range(0, x_len))  # Qtde de Produtos de Suplementação Inteiros
    P_Y = list(range(0, y_len))  # Qtde de Produtos de Suplementação que podem ser fracionados
    N = list(range(0, a))    # Qtde de Nutrientes avaliados

    x = [LpVariable(f"x{i}", lowBound=0, cat='Integer') for i in P_X]  #Variável que determinará quantidade dos produtos não fracionados
    y = [LpVariable(f"y{i}", lowBound=0) for i in P_Y]                 #Variável que determinará quantidade de produtos fracionados
    desvio = [LpVariable(f"desvio{i}", lowBound=0) for i in N]         #Variável que armazena do desvio percentual da quantidade de cada nutriente em relação aos limites SOFT



    Q = {}
    for n in N:
        Q[n] = lpSum(x[p] * matrix_composition_integer[p][n] for p in P_X) + \
            lpSum(y[p] * matrix_composition_fraction[p][n] for p in P_Y)     #Criação do vetor que calcula a quantidade de cada nutriente a depender da cesta escolhida

    modelo_det += lpSum(desvio[n] if soft_lower_limit[n] != "Sem limite" else 0 for n in N), "Minimize desvio condicional"  #Função OBJETIVO

    for n in N:
        modelo_det += Q[n] >= hard_lower_limit[n], f"limite1_{n}" #Restrição de limite INFERIOR HARD
        modelo_det += Q[n] <= hard_upper_limit[n], f"limite2_{n}" #Restrição de limite SUPERIOR HARD
        print(hard_lower_limit[n])
        print(Q[n])

    for n in N:
        if soft_lower_limit[n] != "Sem limite":
            modelo_det += desvio[n] >= ((soft_lower_limit[n] - Q[n]) / soft_lower_limit[n])*peso[n], f"limite_inferior_soft_{n}" #Restrição de limite INFERIOR SOFT
        if soft_upper_limit[n] != "Sem limite":
            modelo_det += desvio[n] >= ((Q[n] - soft_upper_limit[n]) / soft_upper_limit[n])*peso[n], f"limite_superior_soft_{n}" #Restrição de limite SUPERIOR SOFT

    modelo_det.solve()

    fo = pulp.value(modelo_det.objective)

    status = LpStatus[modelo_det.status] # Verifica o status da solução

    values_x = []
    values_y = []
    value_Q  = []
    value_Desvio = []

    # Imprime os valores das variáveis de decisão x e y
    for p in P_X:
        values_x.append(x[p].value())


    for p in P_Y:
        values_y.append(y[p].value())

    for n in N:
        value_Q.append(Q[n].value())

    for n in N:
        value_Desvio.append(desvio[n].value())




    #Construindo Dataframes de OUTPUTS

    df_limits_data['Qtde Nutrientes'] = value_Q
    ordem_limites = ['nutriente','limite_inferior_hard','limite_inferior_soft','Qtde Nutrientes','limite_superior_soft','limite_superior_hard','peso']
    df_limits_data = df_limits_data[ordem_limites]

    df_nutrients_data_result = df_limites_input.copy()

    for idx, row in df_nutrients_data_result.iterrows():
      if "por hora de atividade" in row['Unidade']:
          df_nutrients_data_result.at[idx, "Valor da Cesta"] = value_Q[idx] / duration_activity
          df_nutrients_data_result.at[idx, "Desvio_Soft"] = value_Desvio[idx]
      elif "por litro de água ingerida" in row['Unidade']:
          df_nutrients_data_result.at[idx, "Valor da Cesta"] = value_Q[idx] / recommended_water_intake
          df_nutrients_data_result.at[idx, "Desvio_Soft"] = value_Desvio[idx]
      elif "por kg de peso corporal" in row['Unidade']:
          df_nutrients_data_result.at[idx, "Valor da Cesta"] = value_Q[idx] / body_weight
          df_nutrients_data_result.at[idx, "Desvio_Soft"] = value_Desvio[idx]
    ordem_limites_2 = ['nutriente','Unidade','limite_inferior_hard','limite_inferior_soft','Valor da Cesta','Desvio_Soft','limite_superior_soft','limite_superior_hard','peso']

    df_nutrients_data_result = df_nutrients_data_result[ordem_limites_2]


    df_supplements_integer['Cesta'] = values_x

    df_supplements_fraction['Cesta'] = values_y
    df_supplements_integer = df_supplements_integer[df_supplements_integer["Cesta"] != 0]
    df_supplements_fraction = df_supplements_fraction[df_supplements_fraction["Cesta"] != 0]
    df_plano_final = pd.concat([df_supplements_integer, df_supplements_fraction], axis=0, ignore_index=True)
    columns_plano = ['Referência','Tipo','Marca','Modelo/Sabor','Cesta']
    df_plano_final = df_plano_final[columns_plano]
    df_plano_final = df_plano_final.sort_values(by='Cesta',ascending=False)

    return  df_plano_final, df_nutrients_data_result, fo

def display_side_by_side(df1, title1, df2, title2, output_html='output.html'):
    # Converte os DataFrames para HTML
    df1_html = df1.to_html(index=False)
    df2_html = df2.to_html(index=False)

    # Formatação HTML para exibição lado a lado
    html_str = f'''
    <html>
    <head>
        <title>DataFrames Lado a Lado</title>
        <style>
            .container {{
                display: flex;
                flex-direction: row;
            }}
            .df-container {{
                margin-right: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="df-container">
                <h3>{title1}</h3>
                {df1_html}
            </div>
            <div class="df-container">
                <h3>{title2}</h3>
                {df2_html}
            </div>
        </div>
    </body>
    </html>
    '''

    # Salva o conteúdo HTML em um arquivo
    with open(output_html, 'w') as file:
        file.write(html_str)

    # Abre o arquivo HTML no navegador
    import webbrowser
    webbrowser.open(output_html)
    
def graficonutrientes(ax, df_nutrients_data_result):
    import pandas as pd
    import numpy as np

    # Substituindo "Sem limite" por NaN e convertendo as colunas para numéricas
    df_nutrients_data_result['limite_inferior_hard'] = pd.to_numeric(df_nutrients_data_result['limite_inferior_hard'], errors='coerce')
    df_nutrients_data_result['limite_inferior_soft'] = pd.to_numeric(df_nutrients_data_result['limite_inferior_soft'].replace('Sem limite', np.nan), errors='coerce')
    df_nutrients_data_result['limite_superior_soft'] = pd.to_numeric(df_nutrients_data_result['limite_superior_soft'].replace('Sem limite', np.nan), errors='coerce')
    df_nutrients_data_result['limite_superior_hard'] = pd.to_numeric(df_nutrients_data_result['limite_superior_hard'], errors='coerce')
    df_nutrients_data_result['Valor da Cesta'] = pd.to_numeric(df_nutrients_data_result['Valor da Cesta'], errors='coerce')

    # Normalizando os valores dos indicadores e os limites
    df_nutrients_data_result['Valor Normalizado'] = (df_nutrients_data_result['Valor da Cesta'] - df_nutrients_data_result['limite_inferior_hard']) / (df_nutrients_data_result['limite_superior_hard'] - df_nutrients_data_result['limite_inferior_hard'])
    df_nutrients_data_result['limite_inferior_hard_normalizado'] = 0
    df_nutrients_data_result['limite_inferior_soft_normalizado'] = (df_nutrients_data_result['limite_inferior_soft'] - df_nutrients_data_result['limite_inferior_hard']) / (df_nutrients_data_result['limite_superior_hard'] - df_nutrients_data_result['limite_inferior_hard'])
    df_nutrients_data_result['limite_superior_soft_normalizado'] = (df_nutrients_data_result['limite_superior_soft'] - df_nutrients_data_result['limite_inferior_hard']) / (df_nutrients_data_result['limite_superior_hard'] - df_nutrients_data_result['limite_inferior_hard'])
    df_nutrients_data_result['limite_superior_hard_normalizado'] = 1

    # Criando o gráfico de dispersão no eixo fornecido
    # Adicionando as linhas horizontais limítrofes hard
    ax.hlines(df_nutrients_data_result['limite_inferior_hard_normalizado'].unique(), xmin=-0.5, xmax=len(df_nutrients_data_result)-0.5, colors='red', linestyles='-', linewidth=2, label='Limite Hard')
    ax.hlines(df_nutrients_data_result['limite_superior_hard_normalizado'].unique(), xmin=-0.5, xmax=len(df_nutrients_data_result)-0.5, colors='red', linestyles='-', linewidth=2)

    # Adicionando as linhas horizontais limítrofes soft como pontilhadas em laranja e preenchimento entre elas
    for index, row in df_nutrients_data_result.iterrows():
        if not pd.isna(row['limite_inferior_soft_normalizado']) and not pd.isna(row['limite_superior_soft_normalizado']):
            ax.fill_between([index-0.5, index+0.5], row['limite_inferior_soft_normalizado'], row['limite_superior_soft_normalizado'], color='gray', alpha=0.5)
            ax.hlines(row['limite_inferior_soft_normalizado'], xmin=index-0.5, xmax=index+0.5, colors='gray', linestyles='--', linewidth=4, label=None)  # label=None para não incluir na legenda
            ax.hlines(row['limite_superior_soft_normalizado'], xmin=index-0.5, xmax=index+0.5, colors='gray', linestyles='--', linewidth=4, label=None)  # label=None para não incluir na legenda

            # Adicionando rótulos com os valores soft não normalizados
            ax.text(index + 0.5, row['limite_inferior_soft_normalizado'], f"{row['limite_inferior_soft']:.0f}", ha='left', va='center', fontsize=12, color='gray',fontweight='bold')
            ax.text(index + 0.5, row['limite_superior_soft_normalizado'], f"{row['limite_superior_soft']:.0f}", ha='left', va='center', fontsize=12, color='gray',fontweight='bold')



    # Plotando os valores medidos na cesta como pontos pretos com rótulos dos valores reais
    for index, row in df_nutrients_data_result.iterrows():
        ax.scatter(index, row['Valor Normalizado'], color='black', zorder=3)
        ax.text(index, row['Valor Normalizado'], f"{row['Valor da Cesta']:.1f}", ha='center', va='bottom', fontsize=12)

    # Adicionando rótulos e título
    ax.set_xlabel('Nutrientes')
    ax.set_xticks(range(len(df_nutrients_data_result)))
    ax.set_xticklabels(df_nutrients_data_result['nutriente'], rotation=45)

    # Removendo as linhas de grade
    ax.grid(False)
    # Removendo os labels do eixo y
    ax.set_yticklabels([])


figura, ax = plt.subplots(figsize=(24, 8))


st.write("### Resultados")
df_plano_final, df_nutrients_data_result ,fo = optimal_supplementation()
resultCol1, resultCol2, resultCol3, resultCol4 = st.columns(4)

# Gera os gráficos nos eixos fornecidos
graficonutrientes(ax, df_nutrients_data_result)
st.write(df_plano_final)
st.pyplot(figura)
