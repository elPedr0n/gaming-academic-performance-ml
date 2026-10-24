# Gera as figuras e as estatisticas da EDA do dataset pos-processamento.
# Uso: python eda/gerar_eda.py  (a partir da raiz do repositorio)
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

SEED = 42
RAIZ = Path(__file__).resolve().parent.parent
FIGS = RAIZ / 'eda' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

# Paleta: rampa ordinal azul (D claro -> A escuro), validada para leitura em fundo claro
CONCEITOS = ['A', 'B', 'C', 'D']
CORES = {'D': '#86b6ef', 'C': '#5598e7', 'B': '#256abf', 'A': '#104281'}
TINTA, TINTA2, MUDO, GRADE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': TINTA2, 'axes.titlecolor': TINTA,
    'xtick.color': MUDO, 'ytick.color': MUDO, 'text.color': TINTA,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


def salvar(fig, nome):
    fig.savefig(FIGS / nome)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dados: bruto (para contexto) e pos-processamento (estado final)
# ---------------------------------------------------------------------------
bruto = pd.read_csv(RAIZ / 'Gaming_Academic_Performance_updated.csv')
df = pd.read_csv(RAIZ / 'Gaming_Academic_Performance_limpo.csv')
df['conceito'] = pd.Categorical(df['conceito'], categories=CONCEITOS, ordered=True)
y_ord = df['conceito'].map({'D': 1, 'C': 2, 'B': 3, 'A': 4}).astype(int)

X = df.drop(columns=['conceito'])
NUM = ['age', 'gaming_hours', 'study_hours', 'sleep_hours', 'attendance', 'social_activity',
       'device_usage', 'reaction_time_ms', 'addiction_score']
ONEHOT = [c for c in X.columns if c.startswith(('gender_', 'gaming_genre_'))]

R = {}  # resultados numericos para o relatorio

R['shape'] = list(df.shape)
R['duplicadas_X'] = int(X.duplicated().sum())
R['duplicadas_X_y'] = int(df.duplicated().sum())
R['faltantes'] = int(df.isna().sum().sum())
R['classes'] = {c: int(n) for c, n in df['conceito'].value_counts().reindex(CONCEITOS).items()}
R['n_unicos'] = {c: int(X[c].nunique()) for c in X.columns}

# Estatisticas descritivas (+ assimetria e curtose)
desc = X[NUM + ['stress_level']].describe().T
desc['skew'] = X[NUM + ['stress_level']].skew()
desc['kurt'] = X[NUM + ['stress_level']].kurt()
R['describe'] = desc.round(3).to_dict(orient='index')

# Frequencia das categorias
R['freq_cat'] = {
    'stress_level': X['stress_level'].value_counts().sort_index().to_dict(),
    **{c: int(X[c].sum()) for c in ONEHOT},
}

# Efeitos do pre-processamento
g = bruto['grades'].clip(0, 100)
R['grades_bruto'] = {'acima_100': int((bruto['grades'] > 100).sum()), 'max': float(bruto['grades'].max()),
                     'iguais_100_pos_clip': int((g == 100).sum()), 'iguais_0': int((g == 0).sum()),
                     'A_vindos_de_100': round(float((g == 100).sum() / R['classes']['A']), 3)}
R['addiction_zero_pos_clip'] = int((X['addiction_score'] == 0).sum())

# Coerencia de dominio
soma_h = X['gaming_hours'] + X['study_hours'] + X['sleep_hours']
R['soma_horas'] = {'max': float(soma_h.max()), 'acima_24': int((soma_h > 24).sum())}

# Outliers (regra de 1.5 IQR)
q1, q3 = X[NUM].quantile(0.25), X[NUM].quantile(0.75)
iqr = q3 - q1
R['outliers_iqr'] = {c: int(((X[c] < q1[c] - 1.5 * iqr[c]) | (X[c] > q3[c] + 1.5 * iqr[c])).sum()) for c in NUM}

# Medias por conceito
R['media_por_conceito'] = df.groupby('conceito', observed=True)[NUM + ['stress_level']].mean().round(2).to_dict(orient='index')

# Associacao de cada atributo com o alvo
assoc = {}
for c in NUM + ['stress_level']:
    grupos = [X.loc[df['conceito'] == k, c] for k in CONCEITOS]
    H, p = stats.kruskal(*grupos)
    eps2 = H / (len(X) - 1)  # epsilon^2 (tamanho de efeito do Kruskal-Wallis)
    rho = stats.spearmanr(X[c], y_ord).statistic
    assoc[c] = {'kruskal_H': round(float(H), 1), 'p': float(p), 'epsilon2': round(float(eps2), 3),
                'spearman': round(float(rho), 3)}


def cramer_v(tab):
    chi2, p, _, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(tab.shape) - 1)))


gender = X[[c for c in ONEHOT if c.startswith('gender_')]].idxmax(axis=1).str.replace('gender_', '')
genre = X[[c for c in ONEHOT if c.startswith('gaming_genre_')]].idxmax(axis=1).str.replace('gaming_genre_', '')
categoricas = {'gender': gender, 'gaming_genre': genre, 'stress_level': X['stress_level'].map({1: 'Low', 2: 'Medium', 3: 'High'})}
for nome, serie in categoricas.items():
    chi2, p, v = cramer_v(pd.crosstab(serie, df['conceito']))
    assoc.setdefault(nome, {}).update({'chi2': round(float(chi2), 1), 'p_chi2': float(p), 'cramer_v': round(float(v), 3)})
R['associacao'] = assoc
R['crosstab'] = {n: pd.crosstab(s, df['conceito'], normalize='index').round(3).to_dict(orient='index')
                 for n, s in categoricas.items()}

mi = pd.Series(mutual_info_classif(X, df['conceito'], discrete_features=[c in ONEHOT or c == 'stress_level' for c in X.columns],
                                   random_state=SEED), index=X.columns).sort_values(ascending=False)
R['mutual_info'] = mi.round(4).to_dict()

# Correlacao entre atributos e multicolinearidade
corr = X[NUM + ['stress_level']].corr(method='spearman')
pares = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False)
R['pares_correlacionados'] = {f'{a} x {b}': round(float(v), 3) for (a, b), v in pares.head(8).items()}

Z = StandardScaler().fit_transform(X[NUM + ['stress_level']])
vif = {}
for i, c in enumerate(NUM + ['stress_level']):
    outros = np.delete(Z, i, axis=1)
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), outros], Z[:, i], rcond=None)
    r2 = 1 - ((Z[:, i] - np.c_[np.ones(len(Z)), outros] @ beta) ** 2).sum() / ((Z[:, i] - Z[:, i].mean()) ** 2).sum()
    vif[c] = round(float(1 / (1 - r2)), 2)
R['vif'] = vif

# Projecoes (PCA nao supervisionada, LDA supervisionada) sobre os dados padronizados
Xs = StandardScaler().fit_transform(X)
pca = PCA(random_state=SEED).fit(Xs)
R['pca_var'] = pca.explained_variance_ratio_.round(4).tolist()
R['pca_comp_90'] = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)
pc = pca.transform(Xs)[:, :2]
lda = LinearDiscriminantAnalysis().fit(Xs, df['conceito'])
ld = lda.transform(Xs)[:, :2]
R['lda_var'] = lda.explained_variance_ratio_.round(4).tolist()
R['pca_pc1_cargas'] = pd.Series(pca.components_[0], index=X.columns).round(3).sort_values(key=abs, ascending=False).head(6).to_dict()
R['lda_ld1_cargas'] = pd.Series(lda.scalings_[:, 0], index=X.columns).round(3).sort_values(key=abs, ascending=False).head(6).to_dict()

# Baseline trivial (sempre prever a classe majoritaria)
p_major = R['classes']['D'] / len(df)
R['baseline_majoritaria'] = {'accuracy': round(p_major, 4), 'f1_macro': round((2 * p_major / (1 + p_major)) / 4, 4)}

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
LABEL = {'age': 'Idade (anos)', 'gaming_hours': 'Horas de jogo/dia', 'study_hours': 'Horas de estudo/dia',
         'sleep_hours': 'Horas de sono/dia', 'attendance': 'Frequencia (%)', 'social_activity': 'Atividade social (0-5)',
         'device_usage': 'Uso de dispositivos (h/dia)', 'reaction_time_ms': 'Tempo de reacao (ms)',
         'addiction_score': 'Score de vicio', 'stress_level': 'Nivel de estresse (1-3)'}

# 01 - distribuicao das classes
fig, ax = plt.subplots(figsize=(6, 3.6))
cont = df['conceito'].value_counts().reindex(CONCEITOS)
barras = ax.bar(CONCEITOS, cont.values, color=[CORES[c] for c in CONCEITOS], width=0.6)
for b, n in zip(barras, cont.values):
    ax.text(b.get_x() + b.get_width() / 2, n + 40, f'{n}\n({n / len(df):.1%})', ha='center', va='bottom', color=TINTA2, fontsize=9)
ax.set_ylim(0, cont.max() * 1.2)
ax.set_xlabel('Conceito')
ax.set_ylabel('Numero de estudantes')
ax.set_title('Distribuicao do atributo alvo (n = 8000)')
ax.grid(axis='x', visible=False)
salvar(fig, '01_distribuicao_conceitos.png')

# 02 - origem do alvo: notas (pos-clip) com os limites dos conceitos
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(g, bins=100, color=CORES['B'], edgecolor='#fcfcfb', linewidth=0.3)
for lim in [60, 75, 90]:
    ax.axvline(lim, color=TINTA2, linestyle='--', linewidth=1)
for x0, x1, k in [(0, 60, 'D'), (60, 75, 'C'), (75, 90, 'B'), (90, 100, 'A')]:
    ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.92, k, ha='center', fontsize=12, fontweight='bold', color=TINTA2)
ax.annotate(f'{R["grades_bruto"]["iguais_100_pos_clip"]} notas = 100', xy=(100, (g == 100).sum()),
            xytext=(70, (g == 100).sum() * 0.85), color=TINTA2, fontsize=9, arrowprops=dict(arrowstyle='->', color=MUDO))
ax.set_xlabel('grades (apos limitar ao intervalo [0, 100])')
ax.set_ylabel('Estudantes')
ax.set_title('Nota original e cortes dos conceitos')
salvar(fig, '02_notas_e_cortes.png')

# 03 - histogramas dos atributos numericos
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for ax, c in zip(axes.flat, NUM):
    ax.hist(X[c], bins=40, color=CORES['C'], edgecolor='#fcfcfb', linewidth=0.3)
    ax.set_title(LABEL[c])
    ax.text(0.98, 0.95, f'assim. = {X[c].skew():.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=8, color=TINTA2)
fig.suptitle('Distribuicao dos atributos numericos', fontweight='bold', y=1.0)
fig.tight_layout()
salvar(fig, '03_histogramas.png')

# 04 - boxplots por conceito
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, c in zip(axes.flat, NUM):
    dados = [X.loc[df['conceito'] == k, c] for k in CONCEITOS]
    bp = ax.boxplot(dados, tick_labels=CONCEITOS, patch_artist=True, widths=0.6,
                    medianprops=dict(color='#fcfcfb', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=2, markerfacecolor=MUDO, markeredgecolor='none', alpha=0.5),
                    whiskerprops=dict(color=MUDO), capprops=dict(color=MUDO), boxprops=dict(linewidth=0))
    for patch, k in zip(bp['boxes'], CONCEITOS):
        patch.set_facecolor(CORES[k])
    ax.set_title(LABEL[c], fontsize=10)
    ax.text(0.98, 0.02, f'eps² = {assoc[c]["epsilon2"]:.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=TINTA2)
    ax.grid(axis='x', visible=False)
fig.suptitle('Atributos por conceito (eps² = tamanho de efeito do Kruskal-Wallis)', fontweight='bold', y=1.0)
fig.tight_layout()
salvar(fig, '04_boxplots_por_conceito.png')

# 05 - densidades dos 4 atributos mais associados, por conceito (sobreposicao entre classes)
top4 = sorted(NUM, key=lambda c: assoc[c]['epsilon2'], reverse=True)[:4]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, c in zip(axes, top4):
    grade_x = np.linspace(X[c].min(), X[c].max(), 300)
    for k in CONCEITOS[::-1]:
        v = X.loc[df['conceito'] == k, c]
        dens = stats.gaussian_kde(v)(grade_x)
        ax.plot(grade_x, dens, color=CORES[k], linewidth=2, label=k)
        ax.fill_between(grade_x, dens, color=CORES[k], alpha=0.08)
    ax.set_title(LABEL[c], fontsize=10)
    ax.set_yticks([])
axes[0].legend(title='Conceito', reverse=True)
fig.suptitle('Densidade por conceito: atributos com maior associacao com o alvo', fontweight='bold', y=1.03)
fig.tight_layout()
salvar(fig, '05_densidades_top4.png')

# 06 - atributos categoricos x conceito (barras 100% empilhadas)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
for ax, (nome, serie) in zip(axes, categoricas.items()):
    tab = pd.crosstab(serie, df['conceito'], normalize='index')[CONCEITOS[::-1]]
    if nome == 'stress_level':
        tab = tab.reindex(['Low', 'Medium', 'High'])
    esquerda = np.zeros(len(tab))
    for k in tab.columns:
        ax.barh(range(len(tab)), tab[k], left=esquerda, color=CORES[k], height=0.6, edgecolor='#fcfcfb', linewidth=2, label=k)
        for i, (l, v) in enumerate(zip(esquerda, tab[k])):
            if v > 0.06:
                ax.text(l + v / 2, i, f'{v:.0%}', ha='center', va='center', color='#ffffff' if k != 'D' else TINTA, fontsize=8)
        esquerda += tab[k].values
    ax.set_yticks(range(len(tab)), tab.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title(f'{nome}  (V de Cramer = {assoc[nome]["cramer_v"]:.3f})', fontsize=10)
    ax.grid(axis='y', visible=False)
axes[0].legend(title='Conceito', ncol=4, loc='lower left', bbox_to_anchor=(0, 1.1), reverse=True)
fig.suptitle('Proporcao de conceitos em cada categoria', fontweight='bold', y=1.15)
fig.tight_layout()
salvar(fig, '06_categoricas_por_conceito.png')

# 13 - frequencia das classes de cada atributo categorico (lado a lado, mesmo eixo y)
ORDEM_CAT = {'gender': ['Male', 'Female', 'Other'], 'gaming_genre': ['FPS', 'RPG', 'Casual'],
             'stress_level': ['Low', 'Medium', 'High']}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (nome, serie) in zip(axes, categoricas.items()):
    cont = serie.value_counts().reindex(ORDEM_CAT[nome])
    barras = ax.bar(cont.index, cont.values, color=CORES['B'], width=0.6)
    for b, n in zip(barras, cont.values):
        ax.text(b.get_x() + b.get_width() / 2, n + 40, f'{n}\n({n / len(serie):.1%})', ha='center', va='bottom', color=TINTA2, fontsize=9)
    ax.set_xlabel(nome)
    ax.set_title(nome, fontsize=10)
    ax.grid(axis='x', visible=False)
axes[0].set_ylim(0, max(s.value_counts().max() for s in categoricas.values()) * 1.2)
axes[0].set_ylabel('Numero de estudantes')
fig.suptitle(f'Distribuicao dos atributos categoricos (n = {len(df)})', fontweight='bold', y=1.0)
fig.tight_layout()
salvar(fig, '13_frequencia_categoricas.png')

# 07 - matriz de correlacao de Spearman (atributos + conceito ordinal)
mat = pd.concat([X[NUM + ['stress_level']], y_ord.rename('conceito (D=1..A=4)')], axis=1).corr(method='spearman')
fig, ax = plt.subplots(figsize=(9, 7.5))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(mat)), mat.columns, rotation=45, ha='right')
ax.set_yticks(range(len(mat)), mat.index)
for i in range(len(mat)):
    for j in range(len(mat)):
        ax.text(j, i, f'{mat.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                color='#ffffff' if abs(mat.iloc[i, j]) > 0.6 else TINTA)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04, label='rho de Spearman')
ax.set_title('Correlacao de Spearman entre atributos e com o conceito')
salvar(fig, '07_correlacao_spearman.png')

# 08 - informacao mutua com o alvo
fig, ax = plt.subplots(figsize=(7, 5))
ordem = mi.sort_values()
ax.barh(ordem.index, ordem.values, color=CORES['B'], height=0.6)
for i, v in enumerate(ordem.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8, color=TINTA2)
ax.set_xlabel('Informacao mutua com o conceito (nats)')
ax.set_title('Relevancia de cada atributo para o alvo')
ax.grid(axis='y', visible=False)
salvar(fig, '08_informacao_mutua.png')

# 09 - diferenca de escala entre atributos (justifica a normalizacao)
fig, ax = plt.subplots(figsize=(7, 4.5))
amp = (X[NUM + ['stress_level']].max() - X[NUM + ['stress_level']].min()).sort_values()
ax.barh([LABEL[c] for c in amp.index], amp.values, color=CORES['C'], height=0.6)
ax.set_xscale('log')
for i, v in enumerate(amp.values):
    ax.text(v * 1.08, i, f'{v:.1f}', va='center', fontsize=8, color=TINTA2)
ax.set_xlabel('Amplitude (max - min), escala log')
ax.set_title('Escalas muito diferentes entre atributos')
ax.grid(axis='y', visible=False)
salvar(fig, '09_escalas.png')

# 10 - matriz de dispersao dos 4 atributos mais associados (amostra)
amostra = df.sample(1500, random_state=SEED)
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for i, a in enumerate(top4):
    for j, b in enumerate(top4):
        ax = axes[i, j]
        if i == j:
            for k in CONCEITOS[::-1]:
                ax.hist(amostra.loc[amostra['conceito'] == k, a], bins=25, color=CORES[k], alpha=0.6)
        else:
            for k in CONCEITOS[::-1]:
                s = amostra[amostra['conceito'] == k]
                ax.scatter(s[b], s[a], s=6, color=CORES[k], alpha=0.7, linewidths=0, label=k)
        if i == 3:
            ax.set_xlabel(LABEL[b], fontsize=8)
        if j == 0:
            ax.set_ylabel(LABEL[a], fontsize=8)
        ax.tick_params(labelsize=7)
axes[0, 1].legend(title='Conceito', reverse=True, markerscale=2, loc='upper left', bbox_to_anchor=(0, 1.45), ncol=4)
fig.suptitle('Dispersao entre os 4 atributos mais associados (amostra de 1500)', fontweight='bold', y=1.0)
fig.tight_layout()
salvar(fig, '10_dispersao_top4.png')

# 11 - projecoes PCA e LDA
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
idx = amostra.index
for ax, P, titulo, (vx, vy) in [
    (axes[0], pc, 'PCA (nao supervisionada)', R['pca_var'][:2]),
    (axes[1], ld, 'LDA (supervisionada)', R['lda_var'][:2]),
]:
    for k in CONCEITOS[::-1]:
        m = (df.loc[idx, 'conceito'] == k).values
        ax.scatter(P[idx][m, 0], P[idx][m, 1], s=8, color=CORES[k], alpha=0.75, linewidths=0, label=k)
    ax.set_xlabel(f'Componente 1 ({vx:.1%} da variancia)')
    ax.set_ylabel(f'Componente 2 ({vy:.1%})')
    ax.set_title(titulo)
axes[0].legend(title='Conceito', reverse=True, markerscale=2)
fig.suptitle('Projecao 2D dos dados padronizados (amostra de 1500)', fontweight='bold', y=1.02)
fig.tight_layout()
salvar(fig, '11_projecoes_pca_lda.png')

# 12 - variancia explicada acumulada (PCA)
fig, ax = plt.subplots(figsize=(7, 3.6))
acum = np.cumsum(pca.explained_variance_ratio_)
ax.bar(range(1, len(acum) + 1), pca.explained_variance_ratio_, color=CORES['D'], width=0.6, label='individual')
ax.plot(range(1, len(acum) + 1), acum, color=CORES['A'], linewidth=2, marker='o', markersize=4, label='acumulada')
ax.axhline(0.9, color=MUDO, linestyle='--', linewidth=1)
ax.text(len(acum), 0.92, '90%', ha='right', color=TINTA2, fontsize=8)
ax.set_xlabel('Componente principal')
ax.set_ylabel('Variancia explicada')
ax.set_xticks(range(1, len(acum) + 1))
ax.set_title('PCA: variancia explicada por componente')
ax.legend(loc='center right')
ax.grid(axis='x', visible=False)
salvar(fig, '12_pca_variancia.png')

# Interacao: estresse x horas de estudo (explica o efeito "contraintuitivo" do estresse)
R['estudo_por_estresse'] = X.groupby('stress_level')['study_hours'].mean().round(2).to_dict()
R['corr_estresse_estudo'] = round(float(stats.spearmanr(X['stress_level'], X['study_hours']).statistic), 3)
R['estresse_parcial'] = {}
for faixa, (lo, hi) in {'estudo<4': (0, 4), '4<=estudo<7': (4, 7), 'estudo>=7': (7, 11)}.items():
    m = (X['study_hours'] >= lo) & (X['study_hours'] < hi)
    R['estresse_parcial'][faixa] = round(float(stats.spearmanr(X.loc[m, 'stress_level'], y_ord[m]).statistic), 3)

# stress_level parece derivado de outros atributos: uma arvore rasa o reconstroi quase sem erro
from sklearn.tree import DecisionTreeClassifier, export_text
outros = X.drop(columns=['stress_level'])
arv = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(outros, X['stress_level'])
R['estresse_arvore'] = {'acuracia': round(float(arv.score(outros, X['stress_level'])), 4),
                        'regras': export_text(arv, feature_names=list(outros.columns), show_weights=True)}

# Quanto da nota original e explicado linearmente pelos atributos (diagnostico, sem validacao)
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(X, g)
R['r2_linear_nota'] = round(float(lin.score(X, g)), 4)
R['r2_linear_nota_sem_saturadas'] = round(float(LinearRegression().fit(X[g < 100], g[g < 100]).score(X[g < 100], g[g < 100])), 4)

(RAIZ / 'eda' / 'resultados_eda.json').write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
