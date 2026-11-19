# Facteurs carbone candidats — Archive du 18 novembre 2026

Source : [Ember, Yearly Electricity Data](https://ember-energy.org/data/yearly-electricity-data/), mise à jour du 17 novembre 2026. Attribution et licence annoncée : Ember, CC-BY-4.0.

Statut au moment de l'extraction : données de candidature pour A2-D7. L'auteur a ensuite retenu les deux variantes et les trois contextes pays présentés ; voir le registre courant et l'audit. Les noms `candidate_*` et le statut des métadonnées restent ceux de l'extraction pour préserver la provenance. Ces fichiers ne sont pas des résultats de calcul de labels.

- `ember_generation_yearly_global.csv.gz` : copie complète du CSV source, compression gzip sans changement du contenu décompressé.
- `candidate_factors.csv` : lignes Total generation pour Kenya, Cameroon et South Africa en 2024.
- `candidate_generation_rows.csv` : toutes les lignes de ces trois pays pour 2024, pour vérifier notamment leur composition électrique.
- `ember_electricity_data_methodology.pdf` : méthodologie consultée.
- `metadata.json` : URL, date, empreintes SHA-256, derniers millésimes disponibles et valeurs extraites.

Pour reproduire l'extraction, décompresser le CSV en mémoire, le lire comme CSV UTF-8, retenir les trois pays, la source Total generation et une intensité renseignée, puis prendre la dernière année commune. Les années les plus récentes sont 2025 pour Kenya et South Africa et 2024 pour Cameroon ; l'année commune est donc 2024. Les intensités extraites sont respectivement 83.516, 219.859 et 717.431 gCO2e/kWh.

L'empreinte SHA-256 du CSV décompressé doit être `43db1b75e3ae16eff5fc62a4bfa1f10876efa24af953e0069a939441ea95c616`. Les autres empreintes figurent dans les métadonnées. Une mise à jour de la source doit créer une nouvelle archive et une version explicite, pas écraser celle-ci.

Interprétation et limites : [audit A2-D7](../../../AUDIT_CARBONE_A2_D7.md).
