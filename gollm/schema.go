package gollm

import (
	"reflect"
	"strings"

	"k8s.io/klog/v2"
)

// BuildSchemaFor will build a schema for the given golang type.
// Because this does not have description populated, it is more useful for the response schema than tools/functions.
func BuildSchemaFor(t reflect.Type) *Schema {
	out := &Schema{}

	switch t.Kind() {
	case reflect.String:
		out.Type = TypeString
	case reflect.Struct:
		out.Type = TypeObject
		out.Properties = make(map[string]*Schema)
		numFields := t.NumField()
		required := []string{}
		for i := 0; i < numFields; i++ {
			field := t.Field(i)
			jsonTag := field.Tag.Get("json")
			if jsonTag == "" {
				continue
			}
			if strings.HasSuffix(jsonTag, ",omitempty") {
				jsonTag = strings.TrimSuffix(jsonTag, ",omitempty")
			} else {
				required = append(required, jsonTag)
			}

			fieldType := field.Type

			fieldSchema := BuildSchemaFor(fieldType)
			out.Properties[jsonTag] = fieldSchema
		}

		if len(required) != 0 {
			out.Required = required
		}
	case reflect.Slice:
		out.Type = TypeArray
		out.Items = BuildSchemaFor(t.Elem())
	default:
		klog.Fatalf("unhandled kind %v", t.Kind())
	}

	return out
}
