# Reporte de limpieza y estandarización

```
[quitar_duplicados_exactos] columna='clientes'
  filas: 12048 -> 12000
  nulos: 0 -> 0
  ADVERTENCIA: se eliminaron 48 filas 100% duplicadas (0.40%)
```

```
[quitar_duplicados_exactos] columna='tarjetas'
  filas: 16634 -> 16585
  nulos: 0 -> 0
  ADVERTENCIA: se eliminaron 49 filas 100% duplicadas (0.29%)
```

```
[quitar_duplicados_exactos] columna='transacciones'
  filas: 194173 -> 193015
  nulos: 0 -> 0
  ADVERTENCIA: se eliminaron 1158 filas 100% duplicadas (0.60%)
```

```
[quitar_duplicados_exactos] columna='interacciones_marketing'
  filas: 36000 -> 36000
  nulos: 0 -> 0
```

```
[quitar_duplicados_exactos] columna='catalogo_comercios'
  filas: 42 -> 42
  nulos: 0 -> 0
```

```
[estandarizar_fecha] columna='fecha_nacimiento'
  filas: 12000 -> 12000
  nulos: 0 -> 0
  formatos detectados: {'%Y-%m-%d': 8400, '%d/%m/%Y': 3000, 'unix_epoch_milisegundos': 600}
```

```
[estandarizar_fecha] columna='fecha_nacimiento'
  filas: 12000 -> 12000
  nulos: 0 -> 0
  formatos detectados: {'%Y-%m-%d': 8400, '%d/%m/%Y': 3000, 'unix_epoch_milisegundos': 600}
```

```
[estandarizar_fecha] columna='fecha_emision'
  filas: 16585 -> 16585
  nulos: 8 -> 8
  formatos detectados: {'%Y-%m-%d': 11605, '%d/%m/%Y': 4143, 'unix_epoch_milisegundos': 829, <NA>: 8}
```

```
[estandarizar_fecha] columna='fecha_activacion'
  filas: 16585 -> 16585
  nulos: 2809 -> 2809
  formatos detectados: {'%Y-%m-%d': 9667, '%d/%m/%Y': 3419, <NA>: 2809, 'unix_epoch_milisegundos': 690}
```

```
[estandarizar_fecha] columna='fecha'
  filas: 193015 -> 193015
  nulos: 0 -> 0
  formatos detectados: {'%Y-%m-%d': 135112, '%d/%m/%Y': 48253, 'unix_epoch_milisegundos': 9650}
```

```
[estandarizar_fecha] columna='fecha_contacto'
  filas: 36000 -> 36000
  nulos: 0 -> 0
  formatos detectados: {'%Y-%m-%d': 25200, '%d/%m/%Y': 9000, 'unix_epoch_milisegundos': 1800}
```

```
[normalizar_texto] columna='canal_adquisicion'
  filas: 12000 -> 12000
  nulos: 212 -> 840
  ADVERTENCIA: 628 valores se convirtieron a NA por matchear un placeholder (ej. 'n/a', 'null')
```

```
[normalizar_texto] columna='ciudad'
  filas: 12000 -> 12000
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='estado_cuenta'
  filas: 12000 -> 12000
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='tipo'
  filas: 16585 -> 16585
  nulos: 0 -> 0
```

```
[estandarizar_tipo_fisica_virtual] columna='tipo'
  filas: 16585 -> 16585
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='es_devolucion'
  filas: 193015 -> 193015
  nulos: 0 -> 0
```

```
[estandarizar_binario] columna='es_devolucion'
  filas: 193015 -> 193015
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='tipo_transaccion'
  filas: 193015 -> 193015
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='campana'
  filas: 36000 -> 36000
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='canal'
  filas: 36000 -> 36000
  nulos: 0 -> 0
```

```
[normalizar_texto] columna='respondio'
  filas: 36000 -> 36000
  nulos: 0 -> 2004
  ADVERTENCIA: 2004 valores se convirtieron a NA por matchear un placeholder (ej. 'n/a', 'null')
```

```
[estandarizar_binario] columna='respondio'
  filas: 36000 -> 36000
  nulos: 2004 -> 2004
```

```
[normalizar_texto] columna='categoria'
  filas: 42 -> 42
  nulos: 0 -> 0
```

```
[convertir_a_numerico] columna='monto'
  filas: 193015 -> 193015
  nulos: 0 -> 0
```

```
[estandarizar_formato_cedula] columna='cedula'
  filas: 12000 -> 12000
  nulos: 0 -> 0
```

