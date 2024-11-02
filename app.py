import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_option('deprecation.showPyplotGlobalUse', False)

# Função para exploração do dataset
def exploracao_dataset():
    st.title("Explorador de bases de attrition")

    # Carregando o arquivo CSV
    uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        # Leitura do dataset como DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Exibindo uma amostra do DataFrame
        st.subheader("Amostra do Dataset")
        st.write(df.head())  # Mostra as primeiras linhas do DataFrame

        traducao_colunas = {
            "Age": "Idade",
            "Attrition": "Desligamento",
            "BusinessTravel": "Viagem_de_Negocios",
            "Department": "Departamento",
            "DistanceFromHome": "Distancia_de_Casa",
            "Education": "Educacao",
            "EducationField": "Area_de_Educacao",
            "EmployeeCount": "Contagem_de_Empregados",
            "EmployeeNumber": "Numero_de_Empregado",
            "EnvironmentSatisfaction": "Satisfacao_ambiental",
            "Gender": "Genero",
            "JobInvolvement": "Envolvimento_no_Trabalho",
            "JobLevel": "Nivel_do_Trabalho",
            "JobRole": "Funcao",
            "JobSatisfaction": "Satisfacao_no_Trabalho",
            "MaritalStatus": "Estado_Civil",
            "MonthlyIncome": "Renda_Mensal",
            "NumCompaniesWorked": "Num_Empresas_Trabalhadas",
            "Over18": "Maior_de_18",
            "PercentSalaryHike": "Aumento_Salarial_Percentual",
            "PerformanceRating": "Avaliacao_de_Desempenho",
            "RelationshipSatisfaction": "Satisfacao_no_Relacionamento",
            "StandardHours": "Horas_Padrao",
            "StockOptionLevel": "Nivel_de_Opcoes_de_Acoes",
            "TotalWorkingYears": "Total_de_Anos_Trabalhados",
            "TrainingTimesLastYear": "Treinamentos_no_Ano_Passado",
            "WorkLifeBalance": "Equilibrio_Entre_Vida_e_Trabalho",
            "YearsAtCompany": "Anos_na_Empresa",
            "YearsSinceLastPromotion": "Anos_Desde_a_Ultima_Promocao",
            "YearsWithCurrManager": "Anos_com_o_Atual_Gerente"
        }

        # Renomear as colunas
        df.rename(columns=traducao_colunas, inplace=True)

        # Plotar distribuição das variáveis numéricas
        variaveis_numericas = ['Idade', 'Distancia_de_Casa', 'Educacao', 'Nivel_do_Trabalho', 'Renda_Mensal',
                            'Num_Empresas_Trabalhadas', 'Aumento_Salarial_Percentual', 'Nivel_de_Opcoes_de_Acoes',
                            'Total_de_Anos_Trabalhados', 'Treinamentos_no_Ano_Passado', 'Anos_na_Empresa',
                            'Anos_Desde_a_Ultima_Promocao', 'Anos_com_o_Atual_Gerente']

         # Verifica se há variáveis numéricas antes de plotar
        if len(variaveis_numericas) != None :
            st.subheader("Distribuição das Variáveis Numéricas")
            plt.figure(figsize=(30, 30))
            for i, variavel in enumerate(variaveis_numericas):
                plt.subplot(5, 3, i+1)
                sns.histplot(df[variavel], kde=True)
                plt.title(f'Distribuição de {variavel}')
                plt.tight_layout()
                st.pyplot()  # Renderiza o gráfico no Streamlit

        variaveis_categoricas = ['Desligamento', 'Viagem_de_Negocios', 'Departamento', 'Area_de_Educacao',
                         'Genero', 'Funcao', 'Estado_Civil', 'Maior_de_18']
        
        if len(variaveis_categoricas) != None:
            st.subheader("Distribuição das Variáveis Categóricas")
            plt.figure(figsize=(20, 20))  # Ajusta o tamanho da figura para variáveis categóricas
            for i, variavel in enumerate(variaveis_categoricas):
                plt.subplot(4, 2, i + 1)
                sns.countplot(data=df, x=variavel)
                plt.title(f'Distribuição de {variavel}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot()

        if 'Desligamento' in df.columns:
            st.subheader("Desligamento vs Variáveis Numéricas")
            plt.figure(figsize=(25, 30))  # Tamanho da figura para comparações de Desligamento
            for i, variavel in enumerate(variaveis_numericas):
                plt.subplot(5, 3, i + 1)
                sns.boxplot(data=df, x='Desligamento', y=variavel)
                plt.title(f'Desligamento vs {variavel}')
                plt.tight_layout()
                st.pyplot()  # Renderiza o gráfico no Streamlit
        else:
            st.write("A coluna 'Desligamento' não foi encontrada no dataset para análise.")

        if 'Desligamento' in df.columns:
            st.subheader("Desligamento vs Variáveis Categóricas")
            plt.figure(figsize=(20, 20))  # Tamanho da figura para comparações de Desligamento com variáveis categóricas
            for i, variavel in enumerate(variaveis_categoricas[1:]):  # Excluindo Desligamento
                plt.subplot(4, 2, i + 1)
                sns.countplot(data=df, x=variavel, hue='Desligamento')
                plt.title(f'Desligamento vs {variavel}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot()  # Renderiza o gráfico no Streamlit
        else:
            st.write("A coluna 'Desligamento' não foi encontrada no dataset para análise.")

        # Matriz de correlação
        numerico_df = df.select_dtypes(include=['float64', 'int64'])     # Selecionando apenas colunas numéricas

        # Calcular a matriz de correlação
        st.subheader("Matriz de correlação")
        correlation_matrix = numerico_df.corr()
        plt.figure(figsize=(12, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlação')
        st.pyplot()

# Função para predição (placeholder para implementação futura)
# Dicionário para a tradução de colunas
traducao_colunas = {
    "Age": "Idade",
    "Attrition": "Desligamento",
    "BusinessTravel": "Viagem_de_Negocios",
    "Department": "Departamento",
    "DistanceFromHome": "Distancia_de_Casa",
    "Education": "Educacao",
    "EducationField": "Area_de_Educacao",
    "EmployeeCount": "Contagem_de_Empregados",
    "EmployeeNumber": "Numero_de_Empregado",
    "EnvironmentSatisfaction": "Satisfacao_ambiental",
    "Gender": "Genero",
    "JobInvolvement": "Envolvimento_no_Trabalho",
    "JobLevel": "Nivel_do_Trabalho",
    "JobRole": "Funcao",
    "JobSatisfaction": "Satisfacao_no_Trabalho",
    "MaritalStatus": "Estado_Civil",
    "MonthlyIncome": "Renda_Mensal",
    "NumCompaniesWorked": "Num_Empresas_Trabalhadas",
    "Over18": "Maior_de_18",
    "PercentSalaryHike": "Aumento_Salarial_Percentual",
    "PerformanceRating": "Avaliacao_de_Desempenho",
    "RelationshipSatisfaction": "Satisfacao_no_Relacionamento",
    "StandardHours": "Horas_Padrao",
    "StockOptionLevel": "Nivel_de_Opcoes_de_Acoes",
    "TotalWorkingYears": "Total_de_Anos_Trabalhados",
    "TrainingTimesLastYear": "Treinamentos_no_Ano_Passado",
    "WorkLifeBalance": "Equilibrio_Entre_Vida_e_Trabalho",
    "YearsAtCompany": "Anos_na_Empresa",
    "YearsSinceLastPromotion": "Anos_Desde_a_Ultima_Promocao",
    "YearsWithCurrManager": "Anos_com_o_Atual_Gerente"
}

def predicao():
# Criação do formulário de entrada
    st.title("Previsão de Attrition de Funcionários")

    # Campos para variáveis numéricas
    idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
    distancia_de_casa = st.number_input("Distância de Casa", min_value=0, max_value=30, value=5)
    renda_mensal = st.number_input("Renda Mensal", min_value=0, value=5000)
    num_empresas_trabalhadas = st.number_input("Número de Empresas Trabalhadas", min_value=0, value=1)
    total_anos_trabalhados = st.number_input("Total de Anos Trabalhados", min_value=0, value=5)
    anos_na_empresa = st.number_input("Anos na Empresa", min_value=0, value=3)
    anos_desde_ultima_promocao = st.number_input("Anos Desde a Última Promoção", min_value=0, value=1)
    anos_com_atual_gerente = st.number_input("Anos com o Atual Gerente", min_value=0, value=2)

    # Campos para variáveis categóricas
    educacao = st.selectbox("Educação", options=["Baixa", "Média", "Alta"])
    nivel_do_trabalho = st.selectbox("Nível do Trabalho", options=["1", "2", "3", "4", "5"])
    maior_de_18 = st.selectbox("Maior de 18 anos?", options=["Sim", "Não"])
    viagem_de_negocios = st.selectbox("Viagem de Negócios", options=["Sim", "Não"])
    departamento = st.selectbox("Departamento", options=["Vendas", "TI", "RH", "Financeiro"])
    area_de_educacao = st.selectbox("Área de Educação", options=["Ciências", "Humanas", "Exatas"])
    genero = st.selectbox("Gênero", options=["Masculino", "Feminino"])
    funcao = st.selectbox("Função", options=["Analista", "Gerente", "Coordenador"])
    estado_civil = st.selectbox("Estado Civil", options=["Solteiro", "Casado", "Divorciado"])
    satisfacao_ambiental = st.selectbox("Satisfação Ambiental", options=[1, 2, 3, 4, 5])
    envolvimento_no_trabalho = st.selectbox("Envolvimento no Trabalho", options=[1, 2, 3, 4, 5])
    satisfacao_no_trabalho = st.selectbox("Satisfação no Trabalho", options=[1, 2, 3, 4, 5])
    aumento_salarial_percentual = st.number_input("Aumento Salarial Percentual", min_value=0, value=0)
    avaliacao_de_desempenho = st.selectbox("Avaliação de Desempenho", options=[1, 2, 3, 4, 5])
    satisfacao_no_relacionamento = st.selectbox("Satisfação no Relacionamento", options=[1, 2, 3, 4, 5])
    horas_padrao = st.number_input("Horas Padrão", min_value=0, value=40)
    nivel_de_opcoes_de_acoes = st.selectbox("Nível de Opções de Ações", options=[0, 1, 2, 3])
    treinamentos_no_ano_passado = st.number_input("Treinamentos no Ano Passado", min_value=0, value=0)
    equilibrio_entre_vida_e_trabalho = st.selectbox("Equilíbrio entre Vida e Trabalho", options=[1, 2, 3, 4, 5])

    # Botão para fazer a previsão
    if st.button("Fazer Previsão"):
        # Criar um DataFrame com as entradas
        entrada = pd.DataFrame({
            "Idade": [idade],
            "Distancia_de_Casa": [distancia_de_casa],
            "Renda_Mensal": [renda_mensal],
            "Num_Empresas_Trabalhadas": [num_empresas_trabalhadas],
            "Total_de_Anos_Trabalhados": [total_anos_trabalhados],
            "Anos_na_Empresa": [anos_na_empresa],
            "Anos_Desde_a_Ultima_Promocao": [anos_desde_ultima_promocao],
            "Anos_com_o_Atual_Gerente": [anos_com_atual_gerente],
            "Educacao": [educacao],
            "Nivel_do_Trabalho": [nivel_do_trabalho],
            "Maior_de_18": [1 if maior_de_18 == "Sim" else 0],
            "Viagem_de_Negocios": [1 if viagem_de_negocios == "Sim" else 0],
            "Departamento": [departamento],
            "Area_de_Educacao": [area_de_educacao],
            "Genero": [genero],
            "Funcao": [funcao],
            "Estado_Civil": [estado_civil],
            "Satisfacao_ambiental": [satisfacao_ambiental],
            "Envolvimento_no_Trabalho": [envolvimento_no_trabalho],
            "Satisfacao_no_Trabalho": [satisfacao_no_trabalho],
            "Aumento_Salarial_Percentual": [aumento_salarial_percentual],
            "Avaliacao_de_Desempenho": [avaliacao_de_desempenho],
            "Satisfacao_no_Relacionamento": [satisfacao_no_relacionamento],
            "Horas_Padrao": [horas_padrao],
            "Nivel_de_Opcoes_de_Acoes": [nivel_de_opcoes_de_acoes],
            "Treinamentos_no_Ano_Passado": [treinamentos_no_ano_passado],
            "Equilibrio_Entre_Vida_e_Trabalho": [equilibrio_entre_vida_e_trabalho],
        })
    
        st.title("Oi")
        le = LabelEncoder()
        entrada['Educacao'] = le.fit_transform(entrada['Educacao'])
        entrada['Nivel_do_Trabalho'] = le.fit_transform(entrada['Nivel_do_Trabalho'])
        entrada['Maior_de_18'] = le.fit_transform(entrada['Maior_de_18'])

        entrada = pd.get_dummies(entrada, columns=['Viagem_de_Negocios', 'Departamento', 'Area_de_Educacao', 'Genero', 'Funcao', 'Estado_Civil'], drop_first=True)

        # Criar instância do scaler
        scaler = StandardScaler()

        # Selecionar as colunas numéricas que serão normalizadas
        num_cols = ['Idade', 'Distancia_de_Casa', 'Renda_Mensal', 'Num_Empresas_Trabalhadas',
                    'Total_de_Anos_Trabalhados', 'Anos_na_Empresa', 'Anos_Desde_a_Ultima_Promocao',
                    'Anos_com_o_Atual_Gerente']

        # Aplicar o scaler nas colunas numéricas
        entrada[num_cols] = scaler.fit_transform(entrada[num_cols])

        #model = joblib.load('modelo_attrition.pkl')
        #previsao = model.predict(entrada)
        #st.write(f"A previsão de attrition é: {previsao[0]}")

# Função principal do app
def main():
    st.sidebar.title("Navegação")
    selected_option = st.sidebar.selectbox("Escolha uma opção", ("Exploração de Dataset", "Predição"))

    if selected_option == "Exploração de Dataset":
        exploracao_dataset()
    elif selected_option == "Predição":
        predicao()

if __name__ == "__main__":
    main()
