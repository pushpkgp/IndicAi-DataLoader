{{- define "indic-ai-data-loader.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "indic-ai-data-loader.fullname" -}}
{{- printf "%s" (include "indic-ai-data-loader.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "indic-ai-data-loader.labels" -}}
app.kubernetes.io/name: {{ include "indic-ai-data-loader.name" . }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}