import array
import streamlit as st
import pandas as pd
from datetime import datetime
from pulp import *
from typing import Tuple
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt

import pandas as pd
from pyomo.environ import *
import  pyomo.environ as pyo
import pyomo.contrib.alternative_solutions as aos


def solve_pyomo_model(body_weight, sweat_rate, duration_activity, composicao_nutricional_table, limites_table, num_solutions=7):
    # Cálculos iniciais
    percentual_maximum_weight_loss = 0.02
    maximum_weight_loss = percentual_maximum_weight_loss * body_weight
    total_sweat_loss = sweat_rate * duration_activity
    recommended_water_intake = total_sweat_loss - maximum_weight_loss

    # Separação dos dados de composição nutricional
    comp_nutri_inteiro = composicao_nutricional_table[composicao_nutricional_table['Permite Fração?'] == 'Não'].reset_index(drop=True)
    comp_nutri_fracao = composicao_nutricional_table[composicao_nutricional_table['Permite Fração?'] == 'Sim'].reset_index(drop=True)

    # Matrizes de composição nutricional
    matrix_composition_integer = comp_nutri_inteiro.iloc[:, 4:12].values
    matrix_composition_fraction = comp_nutri_fracao.iloc[:, 4:12].values

    # Processamento dos limites
    def tratar_limite(vetor, duracao_atividade, recommended_water_intake, body_weight):
        for i in range(len(vetor)):
            if i == 0:  # CHO - Carboidrato
                vetor[i] *= duracao_atividade
            elif 1 <= i <= 6:  # Eletrólitos
                if vetor[i] != "Sem limite":
                    vetor[i] *= recommended_water_intake
            elif i == 7:  # Cafeína
                vetor[i] *= body_weight
        return vetor

    limite_inferior_hard = limites_table.iloc[0:8, 2].tolist()
    limite_inferior_soft = limites_table.iloc[0:8, 4].tolist()
    limite_superior_soft = limites_table.iloc[0:8, 5].tolist()
    limite_superior_hard = limites_table.iloc[0:8, 3].tolist()

    limite_inferior_hard_tratado = tratar_limite(limite_inferior_hard, duration_activity, recommended_water_intake, body_weight)
    limite_inferior_soft_tratado = tratar_limite(limite_inferior_soft, duration_activity, recommended_water_intake, body_weight)
    limite_superior_soft_tratado = tratar_limite(limite_superior_soft, duration_activity, recommended_water_intake, body_weight)
    limite_superior_hard_tratado = tratar_limite(limite_superior_hard, duration_activity, recommended_water_intake, body_weight)

    nome_coluna_nutrientes = limites_table.iloc[0:8, 0].tolist()

    # Início do modelo Pyomo
    model = ConcreteModel()

    # Conjuntos
    model.nutrientes = RangeSet(0, len(nome_coluna_nutrientes) - 1)
    model.itens_inteiros = RangeSet(0, len(comp_nutri_inteiro) - 1)
    model.itens_fracionados = RangeSet(0, len(comp_nutri_fracao) - 1)

    # Parâmetros
    model.limite_inferior_hard = Param(model.nutrientes, initialize=dict(enumerate(limite_inferior_hard_tratado)))
    model.limite_inferior_soft = Param(model.nutrientes, initialize=dict(enumerate(limite_inferior_soft_tratado)))
    model.limite_superior_soft = Param(model.nutrientes, initialize=dict(enumerate(limite_superior_soft_tratado)))
    model.limite_superior_hard = Param(model.nutrientes, initialize=dict(enumerate(limite_superior_hard_tratado)))

    # Matrizes de composição
    matrix_integer = {(i, n): matrix_composition_integer[i][n] for i in range(len(comp_nutri_inteiro)) for n in range(len(nome_coluna_nutrientes))}
    matrix_fraction = {(i, n): matrix_composition_fraction[i][n] for i in range(len(comp_nutri_fracao)) for n in range(len(nome_coluna_nutrientes))}

    # Variáveis de decisão
    model.x = Var(model.itens_inteiros, domain=NonNegativeIntegers)  # Inteiros
    model.y = Var(model.itens_fracionados, domain=NonNegativeReals)  # Fracionários
    model.desvio = Var(model.nutrientes, domain=NonNegativeReals)  # Desvios

    # Função objetivo
    def objetivo(model):
        return sum(
            model.desvio[n]
            for n in model.nutrientes
            if limite_inferior_soft_tratado[n] != "Sem limite"
        )
    model.objetivo = Objective(rule=objetivo, sense=minimize)

    # Expressão Q (composição dos nutrientes)
    def regra_Q(model, n):
        return sum(model.x[i] * matrix_integer[i, n] for i in model.itens_inteiros) + \
               sum(model.y[j] * matrix_fraction[j, n] for j in model.itens_fracionados)
    model.Q = Expression(model.nutrientes, rule=regra_Q)

    # Restrições hard
    model.restricoes_inferiores = Constraint(
        model.nutrientes,
        rule=lambda model, n: model.Q[n] >= model.limite_inferior_hard[n]
    )
    model.restricoes_superiores = Constraint(
        model.nutrientes,
        rule=lambda model, n: model.Q[n] <= model.limite_superior_hard[n]
    )

    # Restrições soft
    model.desvio_inferior = Constraint(
        model.nutrientes,
        rule=lambda model, n: model.desvio[n] >= (model.limite_inferior_soft[n] - model.Q[n]) /(model.limite_inferior_soft[n]+0.00000001)
        if model.limite_inferior_soft[n] != "Sem limite" else Constraint.Skip
    )
    model.desvio_superior = Constraint(
        model.nutrientes,
        rule=lambda model, n: model.desvio[n] >= (model.Q[n] - model.limite_superior_soft[n]) / (model.limite_superior_soft[n]+0.0000001)
        if model.limite_superior_soft[n] != "Sem limite" else Constraint.Skip
    )

    # Restrições de inclusão obrigatória
    for idx, row in comp_nutri_inteiro.iterrows():
        if row["Quero forçar inclusão na cesta?"] == "Sim":
            quantidade_forcada = row["Em qual quantidade?"]
            model.add_component(f"forcar_inclusao_{idx}", Constraint(expr=model.x[idx] == quantidade_forcada))

    # Resolver o modelo
    solver = SolverFactory('gurobi')
    solver.set_executable(r'C:\\Users\\luiza\\Desktop\\glpk-4.65\\w64\\glpsol.exe')

    # Gerar múltiplas soluções usando AOS
    solns = aos.gurobi_generate_solutions(model, num_solutions=num_solutions)

    # Processar soluções
    df_solucoes_plano_final = []
    df_solucoes_nutrientes = []

    for sol in solns:
        # Converter solução para dicionário
        solution_data = sol.to_dict()["solution"]
        objetivo_value = sol.to_dict()["objective_value"]

        # Separar x_values e y_values
        x_values = {int(k.strip("x[]")): v for k, v in solution_data.items() if k.startswith("x[")}
        y_values = {int(k.strip("y[]")): v for k, v in solution_data.items() if k.startswith("y[")}

        # Calcular valores de Q para cada nutriente
        Q_values = [
            sum(x_values.get(i, 0) * matrix_composition_integer[i][n] for i in range(len(comp_nutri_inteiro))) +
            sum(y_values.get(j, 0) * matrix_composition_fraction[j][n] for j in range(len(comp_nutri_fracao)))
            for n in range(len(nome_coluna_nutrientes))
        ]

        # Criar DataFrame com limites e valores de Q
        df_solucao_nutrientes = pd.DataFrame({
            'Nutriente': nome_coluna_nutrientes,
            'Limite Inferior Hard': limite_inferior_hard_tratado,
            'Limite Inferior Soft': limite_inferior_soft_tratado,
            'Q': Q_values,
            'Limite Superior Soft': limite_superior_soft_tratado,
            'Limite Superior Hard': limite_superior_hard_tratado,
        })
        df_solucoes_nutrientes.append(df_solucao_nutrientes)

        # Preparar DataFrames para variáveis inteiras e fracionárias
        df_prod_inteiro = pd.DataFrame.from_dict(x_values, orient="index", columns=["Posicao"]).reset_index()
        df_prod_inteiro.columns = ["Index", "Posicao"]
        df_prod_inteiro = df_prod_inteiro[df_prod_inteiro["Posicao"] > 0]

        df_prod_fracao = pd.DataFrame.from_dict(y_values, orient="index", columns=["Posicao"]).reset_index()
        df_prod_fracao.columns = ["Index", "Posicao"]
        df_prod_fracao = df_prod_fracao[df_prod_fracao["Posicao"] > 0]

        # Concatenar resultados
        df_inteiro = pd.concat([comp_nutri_inteiro, df_prod_inteiro.set_index("Index")], axis=1).dropna(subset=["Posicao"])
        df_fracao = pd.concat([comp_nutri_fracao, df_prod_fracao.set_index("Index")], axis=1).dropna(subset=["Posicao"])

        df_plano_final = pd.concat([df_inteiro, df_fracao], axis=0, ignore_index=True)
        df_solucoes_plano_final.append(df_plano_final)

    # Ajustar valores de Q
    for df_solucao in df_solucoes_nutrientes:
        df_solucao.loc[0, "Q"] = (df_solucao.loc[0,"Q"] / duration_activity).round(1)  # Carboidrato (g/hora)
        df_solucao.loc[1:6, "Q"] = (df_solucao.loc[1:6, "Q"] / recommended_water_intake).round(1)  # Nutrientes baseados em mg/L
        df_solucao.loc[7, "Q"] = (df_solucao.loc[7, "Q"] / body_weight).round(1)  # Cafeína (mg/kg)

    # Criar o dataframe df_nutrientes_resumo com os dados da limites_table
    df_nutrientes_resumo = limites_table.copy()

    # Adicionar colunas com os valores ajustados de Q de cada solução
    for i, df_solucao in enumerate(df_solucoes_nutrientes):
        df_nutrientes_resumo[f"Solução {i+1}"] = df_solucao["Q"]

    # Selecionar apenas as quatro primeiras colunas de composicao_nutricional_table
    composicao_nutricional_basico = composicao_nutricional_table[["Referência", "Tipo", "Marca", "Modelo/Sabor"]]

    # Criar df_solucoes_resumo baseado nas colunas selecionadas
    df_solucoes_resumo = composicao_nutricional_basico.copy()

    for idx, solucao_df in enumerate(df_solucoes_plano_final):
        df_solucoes_resumo[f'Solução {idx + 1}'] = df_solucoes_resumo['Referência'].map(
            solucao_df.set_index('Referência')['Posicao']
        )

    df_solucoes_resumo = df_solucoes_resumo.fillna('')

    return recommended_water_intake,df_nutrientes_resumo, df_solucoes_resumo


# Cabeçalho
st.write("## Calculadora de Reposição de Nutrientes")

# Criação das abas
tab1, tab2, tab3, tab4 = st.tabs([
    "Dados do Paciente", 
    "Tabela de Reposição de Nutrientes", 
    "Suplementos Disponíveis", 
    "Cálculo e Resultados"
])

# Aba 1: Dados do Paciente
with tab1:
    st.write("### Dados do Paciente")
    pacCol1, pacCol2 = st.columns(2)

    # Inputs do usuário
    body_weight = pacCol1.number_input("Peso do Atleta (Kg)",value=63)
    duration_activity = pacCol1.number_input("Duração prevista da atividade (h)", value=3.0)
    percent_max_weight_loss = pacCol2.number_input("Perda máxima de peso (Valor padrão:2%)", value=0.02)
    sweat_rate = pacCol2.number_input("Taxa de Sudorese (L/h)",value=1.4)

    # Botão para salvar os dados
    if st.button("Salvar Dados do Paciente", key="save_paciente"):
        st.session_state.saved_paciente_data = {
            "Peso do Atleta (Kg)": body_weight,
            "Duração prevista da atividade (h)": duration_activity,
            "Perda máxima de peso (%)": percent_max_weight_loss,
            "Taxa de Sudorese (L/h)": sweat_rate,
        }
        st.success("Dados do paciente salvos com sucesso!")

    # Exibe os dados salvos (opcional)
    if "saved_paciente_data" in st.session_state:
        st.write("### Dados Salvos:")
        st.write(st.session_state.saved_paciente_data)

# Aba 2: Tabela de Reposição de Nutrientes
with tab2:
    st.write("### Tabela de Reposição de Nutrientes")
    try:
        excel_file = "Input_Dados.xlsx"  # Caminho para o arquivo Excel
        sheet_name = "Limites"  # Nome da aba
        limites_table = pd.read_excel(excel_file, sheet_name=sheet_name,usecols="A:F",nrows=10)
    except Exception as e:
        st.error("Erro ao carregar os dados do Excel. Certifique-se de que o arquivo e a aba existem.")
        st.stop()

    # Tabela editável
    limites_table = st.data_editor(limites_table, use_container_width=True)

    # Botão para salvar as alterações
    if st.button("Salvar Alterações na Tabela de Nutrientes", key="save_limites"):
        st.session_state.saved_limites_table = limites_table
        st.success("Tabela de Reposição de Nutrientes salva com sucesso!")

    # Exibe os dados salvos (opcional)
    if "saved_limites_table" in st.session_state:
        st.write("### Tabela Salva:")
        st.write(st.session_state.saved_limites_table)

# Aba 3: Suplementos Disponíveis
with tab3:
    st.write("### Suplementos Disponíveis")
    try:
        # Carregar dados da aba "composicaonutricional" do Excel
        composicao_nutricional_table = pd.read_excel(
            "Input_Dados.xlsx",
            sheet_name="composicaonutricional",
            usecols="A:P",  # Ajuste para as colunas desejadas
            nrows=20  # Número inicial de linhas
        )
    except Exception as e:
        st.error("Erro ao carregar os dados do Excel. Certifique-se de que o arquivo e a aba existem.")
        st.stop()

    # Tabela editável
    composicao_nutricional_table = st.data_editor(
        composicao_nutricional_table,
        use_container_width=True,
        num_rows="dynamic"  # Permite adicionar novas linhas dinamicamente
    )

    # Botão para salvar as alterações
    if st.button("Salvar Alterações nos Suplementos", key="save_composicao"):
        st.session_state.saved_composicao_nutricional_table = composicao_nutricional_table
        st.success("Tabela de Suplementos salva com sucesso!")

    # Exibe os dados salvos (opcional)
    if "saved_composicao_nutricional_table" in st.session_state:
        st.write("### Tabela Salva:")
        st.write(st.session_state.saved_composicao_nutricional_table)

# Aba 4: Cálculo e Resultados
with tab4:
    st.write("### Cálculo e Resultados")

    if st.button("Calcular", key="calcular_resultados"):
        st.write("### Resultados")

        # Verifica se os dados necessários estão salvos
        if "saved_paciente_data" in st.session_state and \
           "saved_limites_table" in st.session_state and \
           "saved_composicao_nutricional_table" in st.session_state:
            
            # Chamada da função solve_pyomo_model com os dados salvos
            try:
                water_intake,df_nutrientes_resumo, df_solucoes_resumo = solve_pyomo_model(
                    body_weight=st.session_state.saved_paciente_data["Peso do Atleta (Kg)"],
                    sweat_rate=st.session_state.saved_paciente_data["Taxa de Sudorese (L/h)"],
                    duration_activity=st.session_state.saved_paciente_data["Duração prevista da atividade (h)"],
                    composicao_nutricional_table=st.session_state.saved_composicao_nutricional_table,
                    limites_table=st.session_state.saved_limites_table,
                    num_solutions=7
                )

                st.write("#### Volume de água a ser ingerida")
                st.markdown(
                    f"""
                        <div style="font-size:24px; color:black;">
                        {round(water_intake, 3)} Litros
                        </div>
                             """,
                      unsafe_allow_html=True
                            )

                st.write("#### Resumo de Soluções")
                st.dataframe(df_solucoes_resumo, use_container_width=True, height=None)  # Exibe tabela completa
                
                # Exibindo os resultados em tabelas completas, uma abaixo da outra
                st.write("#### Resumo de Nutrientes")
                st.dataframe(df_nutrientes_resumo, use_container_width=True, height=None)  # Exibe tabela completa

            

            except Exception as e:
                st.error(f"Erro ao realizar o cálculo: {e}")
                st.write("Detalhes do erro:")
                st.exception(e)
        else:
            st.error("Por favor, salve os dados nas abas anteriores antes de calcular.")





