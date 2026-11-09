# Supplementary dataset: Subcritical alkaline delignification of oil palm empty fruit bunches

## Description

This deposit contains the experimental dataset and derived files supporting the statistical
modeling, response-surface analysis, prediction comparison, and desirability-based
optimization reported in the associated research article on subcritical alkaline
delignification of oil palm empty fruit bunches (EFB).

The experimental design comprises 128 independent experimental runs: 64 combinations
of operating conditions, each performed in duplicate. The factors were reaction
temperature (100, 120, 140, and 160 °C), NaOH loading (10 and 20 wt.% relative to
initial dry biomass), and reaction time (10–80 min at 10-min intervals).

The response DeligYield is the fraction of initial lignin removed, expressed as a
dimensionless value between 0 and 1. Percentage values in files whose column names
end in `_pct` are expressed as percent (0–100).

## Files

1. `datos_originales.csv` — 128 experimental runs, including operating conditions,
   measured/derived lignin-removal response, block information, and coded variables
   used in the statistical analysis. This is the analysis dataset supplied by the
   authors; it contains both experimental and derived columns, not only unprocessed
   instrument readings.

2. `comparacion_predicciones.csv` — observed delignification yields, predictions from
   the linear RSM and beta-regression models, and corresponding prediction errors.
   Rows correspond in order to the 128 rows in `datos_originales.csv`; the
   `Observado` column matches `DeligYield` row by row.

3. `metricas_modelos_finales.csv` — summary prediction metrics (RMSE, MAE, and
   correlation) for the linear RSM and beta-regression models. The response metrics
   are expressed on the fractional 0–1 scale.

4. `superficies_respuesta_beta.csv` — beta-regression model predictions over a
   temperature–NaOH loading grid at reaction times of 10, 45, and 80 min. These
   are model-generated predictions, not additional experimental observations.
   Intermediate NaOH loadings are interpolated because the experimental design
   included only 10 and 20 wt.% NaOH.

5. `malla_deseabilidad_figura.csv` — prediction and desirability grid used to
   generate the application-oriented desirability figure at 10 min. Columns include
   predicted delignification yield and individual/global desirability components.

6. `optimos_figura.csv` — selected operating conditions, predicted delignification
   yield, and desirability values for the three application categories considered:
   packaging and corrugated board; sack and wrapping paper; and printing and writing
   paper. These are model-based candidate conditions and have not been independently
   validated as commercial pulp specifications.

7. `analisis_sensibilidad_deseabilidad.csv` — sensitivity-analysis results for
   alternative desirability exponents and weighting scenarios. The file contains
   alternative optimized conditions and global desirability values.

## Variable definitions and units

- `BLOQUE`, `Bloque_F`: experimental block identifier (1 or 2).
- `TEMPERATURA`, `T_C`: reaction temperature in °C.
- `T_Abs`, `T_Abs_F`: absolute temperature in K, as stored in the supplied analysis file.
- `CARGA`, `Carga_F`: NaOH loading in wt.% relative to initial dry EFB.
- `TIEMPO`, `Tiempo_F`: reaction time in min.
- `LIGNINA REMOVIDA (g)`: mass of lignin removed, in g, as named in the supplied file.
- `Baseload`: initial lignin mass used as the denominator for calculating delignification yield, in g (approximately 4.6053 g per run).
- `DeligYield`: fraction of initial lignin removed (dimensionless, 0–1).
- `T_c`, `C_c`, `t_c`: coded temperature, NaOH loading, and reaction time, respectively.
- `DeligYield_pred`: model-predicted delignification yield as a fraction (0–1).
- `DeligYield_pct`: model-predicted delignification yield in percent (0–100).
- `RMSE`: root mean square error.
- `MAE`: mean absolute error.
- `AIC`: Akaike information criterion (where reported in the article).
- `BIC`: Bayesian information criterion (where reported in the article).
- `D_global` / `Deseabilidad`: global desirability index (dimensionless, 0–1).

## Important notes

- The response-surface and desirability-grid files contain model predictions, not new
  experimental measurements.
- The experimental design tested only two NaOH loadings (10 and 20 wt.%). Values at
  intermediate loadings in prediction grids are model interpolations within the
  studied domain.
- The optimum conditions are outputs of the selected statistical/desirability
  framework and should be interpreted as candidate operating conditions requiring
  experimental validation. They do not by themselves demonstrate final pulp quality
  or compliance with commercial paper-grade specifications.
- The supplied `datos_originales.csv` includes a column named `RESIDUOS`. Its exact
  definition is not stated in the file itself and should be clarified by the authors
  in the article or in a revised data dictionary before final publication.
- CSV files are provided as supplied by the authors. Numeric decimal separators are
  periods. The original Spanish column names have been retained to preserve the
  supplied dataset.

## Suggested citation

Please cite the associated research article when using these data. The final article
citation and DOI should be added to this record after publication.
