| periode    | modele              | MAE (MW)   |   MAPE (%) | Gain vs naïf   | Écart vs RTE J-1   |
|:-----------|:--------------------|:-----------|-----------:|:---------------|:-------------------|
| validation | naif_saisonnier     | 3 384      |       6.34 | +0.0 %         | +158.7 %           |
| validation | rte_j1              | 1 308      |       2.63 | +61.3 %        | +0.0 %             |
| validation | rte_j1_recalee      | 914        |       1.8  | +73.0 %        | -30.1 %            |
| validation | regression_lineaire | 1 300      |       2.6  | +61.6 %        | -0.6 %             |
| validation | lightgbm            | 1 204      |       2.3  | +64.4 %        | -8.0 %             |
| validation | lstm                | 1 089      |       2.09 | +67.8 %        | -16.7 %            |
| validation | combinaison         | 1 015      |       1.94 | +70.0 %        | -22.4 %            |
| test       | naif_saisonnier     | 3 499      |       6.54 | +0.0 %         | +176.3 %           |
| test       | rte_j1              | 1 267      |       2.49 | +63.8 %        | +0.0 %             |
| test       | rte_j1_recalee      | 853        |       1.68 | +75.6 %        | -32.6 %            |
| test       | regression_lineaire | 1 329      |       2.67 | +62.0 %        | +4.9 %             |
| test       | lightgbm            | 1 185      |       2.28 | +66.1 %        | -6.5 %             |
| test       | lstm                | 1 091      |       2.09 | +68.8 %        | -13.9 %            |
| test       | combinaison         | 1 024      |       1.96 | +70.7 %        | -19.2 %            |