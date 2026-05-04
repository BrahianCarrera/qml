import json

notebook_path = "c:/Users/Usuario/Documents/qml/wine_tsne.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Markdown PCA
cell_pca_md = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "## 6. Reducci\u00f3n con PCA para Angle Encoding (Reduciendo Qubits a 4)\n",
    "Podemos utilizar An\u00e1lisis de Componentes Principales (PCA) para reducir a 4 variables primero y evaluarlo con Angle Encoding. Esto disminuye la cantidad de qubits requeridos de 13 a 4."
  ]
}

# Cell 2: Code PCA
cell_pca_code = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "from sklearn.decomposition import PCA\n",
    "\n",
    "pca = PCA(n_components=4)\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "print(f\"Varianza retenida al reducir a 4 componentes: {pca.explained_variance_ratio_.sum():.2%}\")\n",
    "\n",
    "resultados_pca = []\n",
    "config_pca = [('ry', 'StandardScaler'), ('ry', 'Robust Scaler'), ('ux', 'StandardScaler'), ('ux', 'Robust Scaler')]\n",
    "\n",
    "for enc, s_name in config_pca:\n",
    "    res = evaluate_qtsne(X_pca, enc, s_name)\n",
    "    resultados_pca.append(res)\n",
    "    \n",
    "df_resultados_pca = pd.DataFrame(resultados_pca)\n",
    "display(df_resultados_pca)\n",
    "\n",
    "df_resultados = pd.concat([df_resultados, df_resultados_pca], ignore_index=True)"
  ]
}

# Cell 3: Markdown Best
cell_best_md = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "## 7. Visualizaci\u00f3n del Mejor Modelo y Circuito Empleado\n",
    "A continuaci\u00f3n ordenamos los modelos evaluados por su MSE para escoger el m\u00e1s estable, mostrar su gr\u00e1fico de separaci\u00f3n y observar la estructura del circuito cu\u00e1ntico utilizado en Qiskit."
  ]
}

# Cell 4: Code Best
cell_best_code = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "from qiskit import QuantumCircuit\n",
    "\n",
    "df_resultados = df_resultados.sort_values(by=\"MSE\", ascending=True)\n",
    "best_config = df_resultados.iloc[0]\n",
    "print(\"La mejor configuraci\u00f3n identificada es:\")\n",
    "display(best_config.to_frame().T)\n",
    "\n",
    "is_pca = best_config[\"Caracter\u00edsticas\"] < 13\n",
    "data_to_encode = X_pca if is_pca else X\n",
    "scaler = StandardScaler() if best_config[\"Scaler\"] == 'StandardScaler' else RobustScaler()\n",
    "data_sc = scaler.fit_transform(data_to_encode)\n",
    "\n",
    "best_method = best_config[\"Configuraci\u00f3n Cuantica\"]\n",
    "\n",
    "if \"Amplitude\" in best_method:\n",
    "    states, n_qubits, depth = get_amplitude_encoded_states(data_sc)\n",
    "    qc = QuantumCircuit(n_qubits)\n",
    "    amp = data_sc[0]\n",
    "    amp = amp / np.linalg.norm(amp)\n",
    "    if len(amp) < 2**n_qubits:\n",
    "        padded_amp = np.zeros(2**n_qubits)\n",
    "        padded_amp[:len(amp)] = amp\n",
    "        amp = padded_amp\n",
    "    qc.initialize(amp, qc.qubits)\n",
    "elif \"RY\" in best_method:\n",
    "    states, n_qubits, depth = get_angle_encoded_states(data_sc, gate='ry')\n",
    "    qc = QuantumCircuit(n_qubits)\n",
    "    for i in range(n_qubits):\n",
    "        qc.ry(data_sc[0, i] if i < data_sc.shape[1] else 0.0, i)\n",
    "elif \"UX\" in best_method:\n",
    "    states, n_qubits, depth = get_angle_encoded_states(data_sc, gate='ux')\n",
    "    qc = QuantumCircuit(n_qubits)\n",
    "    for i in range(n_qubits):\n",
    "        qc.rx(data_sc[0, i] if i < data_sc.shape[1] else 0.0, i)\n",
    "\n",
    "features = np.abs(states)\n",
    "tsne_best = TSNE(n_components=2, random_state=42, perplexity=30)\n",
    "X_emb_best = tsne_best.fit_transform(features)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(x=X_emb_best[:,0], y=X_emb_best[:,1], hue=target_names[y], palette='viridis', s=100)\n",
    "plt.title(f\"Mejor Quantum t-SNE Clustering - {best_method} con {n_qubits} Qubits\")\n",
    "plt.xlabel('Dim 1')\n",
    "plt.ylabel('Dim 2')\n",
    "plt.legend(title='Tipos de Vino')\n",
    "plt.show()\n",
    "\n",
    "print(\"Estructura del circuito utilizado para la primera muestra:\")\n",
    "display(qc.draw('mpl'))"
  ]
}

nb["cells"].extend([cell_pca_md, cell_pca_code, cell_best_md, cell_best_code])

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
