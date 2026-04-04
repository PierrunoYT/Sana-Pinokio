module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: { },
        // cwd = project root so venv resolves to ./env (not app/env when path was "app")
        message: [
          "python app/app.py --port {{port}}",
        ],
        on: [{
          // Capture localhost URL in Gepeto-recommended generic form
          "event": "/(http:\\/\\/[0-9.:]+)/",
          "done": true
        }]
      }
    },
    {
      // Set the local 'url' variable for pinokio.js to display "Open WebUI"
      method: "local.set",
      params: {
        // input.event[1] captures the URL from the regex capture group
        url: "{{input.event[1]}}"
      }
    },
  ]
}
