# Subir a GitHub — paso a paso (repo privado: mouseosc-pipeline)

## ⚠ Por qué NO usamos git dentro de iCloud

El proyecto está en una carpeta de **iCloud Drive**. iCloud sincroniza la carpeta
oculta `.git` y deja archivos de bloqueo pegados que rompen git. Por eso primero
copiamos el proyecto a una carpeta local (fuera de iCloud) y ahí trabajamos git.

---

## Opción A — con GitHub CLI `gh` (la más rápida)

Si tienes `gh` instalado (compruébalo con `gh --version`). Pega esto en la
**Terminal** del Mac, una línea a la vez:

```bash
# 1. Copiar el proyecto fuera de iCloud
mkdir -p ~/Code
cp -R "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Antigravity/AnalisisAleSteph/PipelineRatonGenerico" ~/Code/mouseosc-pipeline
cd ~/Code/mouseosc-pipeline

# 2. Limpiar cualquier .git trabado que se haya copiado
rm -rf .git

# 3. Iniciar git y primer commit
git init
git add -A
git commit -m "Pipeline genérico mouseosc v0.1"

# 4. Crear el repo PRIVADO en GitHub y subir (gh lo hace todo)
gh repo create mouseosc-pipeline --private --source=. --push
```

Si `gh` no está instalado:  `brew install gh && gh auth login`  y repite el paso 4.

---

## Opción B — sin `gh` (con la web de GitHub)

```bash
# pasos 1–3 iguales que arriba (copiar, rm -rf .git, init, add, commit)
```

Luego, en el navegador: entra a https://github.com/new , crea un repositorio
**privado** llamado `mouseosc-pipeline`, **sin** marcar "Add a README". GitHub te
mostrará la URL. Vuelve a la Terminal:

```bash
git remote add origin https://github.com/<TU-USUARIO>/mouseosc-pipeline.git
git branch -M main
git push -u origin main
```

Te pedirá usuario y un **token** (no la contraseña): créalo en
https://github.com/settings/tokens (tipo "classic", permiso `repo`).

---

## Después

- Para subir cambios futuros:  `git add -A && git commit -m "mensaje" && git push`
- El `.gitignore` ya excluye entornos, cachés, resultados y datos crudos: solo se
  versiona el código y la configuración.
- Recuerda: edita y corre git en `~/Code/mouseosc-pipeline` (local), no en la
  copia de iCloud.
