from logica.simulacion import ejecutar_con_genetico

casos = [
    ("Extremo (seco/caliente)", 10.0, 38.0, 25.0, 1800.0),
    ("Intermedio clasico",      35.0, 25.0, 60.0, 1000.0),
    ("Baja demanda",            58.0, 14.0, 90.0, 250.0),
    ("Seco + temp media",       18.0, 24.0, 55.0, 900.0),
]

for nombre, hs, t, hr, par in casos:
    print(f"\n{'='*55}")
    print(f" {nombre}")
    print(f" hs={hs}, t={t}, hr={hr}, par={par}")
    print(f"{'='*55}")
    r = ejecutar_con_genetico(hs, t, hr, par, n_generaciones=50, tam_poblacion=30)
    sin_o = r["resultado_sin_optimizar"]
    opt_o = r["resultado_optimizado"]
    fin_o = r["resultado_final"]
    print(f"  Base:      {sin_o['agua']:.2f} min  ({sin_o['nivel']})")
    print(f"  Propuesta: {opt_o['agua']:.2f} min  ({opt_o['nivel']})")
    print(f"  Final:     {fin_o['agua']:.2f} min  ({fin_o['nivel']})")
    diff = opt_o['agua'] - sin_o['agua']
    print(f"  Diferencia: {diff:+.2f} min")
    print(f"  Fitness:   {r['mejor_fitness']:.4f}")
    print(f"  Aceptada:  {r['se_acepta_optimizacion']}")
    print(f"  Motivo:    {r['motivo_decision']}")
