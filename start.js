module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        message: [
          "python app.py --port {{port}}",
        ],
        on: [{
          // Monitor for the HTTP URL pattern
          "event": "/(http:\\/\\/[0-9.]+:[0-9]+)/",
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
