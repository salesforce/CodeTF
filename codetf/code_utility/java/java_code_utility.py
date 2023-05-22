from codetf.code_utility.language_specific_utility import LanguageSpecificUtility

class JavaCodeUtility(LanguageSpecificUtility):
    def __init__(self):
        super(JavaCodeUtility, self).__init__('java')
        self.var_node_types = {'identifier'}
        self.var_filter_types = {'class_declaration', 'method_declaration', 'method_invocation'}

    def get_identifier_nodes(self, tree, text):
        var_nodes = []
        var_renames = {}
        class_count, method_count = 1, 1
        queue = [tree.root_node]
        while queue:
            current_node = queue.pop(0)
            current_type = str(current_node.type)
            if current_type == "class_declaration":
                class_name_node = current_node.children[1]  # Assuming class name is the first child after 'class' keyword
                var_nodes.append([class_name_node, "class_{}".format(class_count)])
                class_count += 1
            elif current_type == "method_declaration":
                method_name_node = current_node.children[1].children[0]  # Assuming method name is the first child of the method declarator
                var_nodes.append([method_name_node, "method_{}".format(method_count)])
                method_count += 1
            for child_node in current_node.children:
                child_type = str(child_node.type)
                if child_type in self.var_node_types:  # only identifier node
                    if current_type in self.var_filter_types:
                        # filter out class/method name or function call identifier
                        continue
                    var_name = text[child_node.start_byte: child_node.end_byte]
                    if var_name not in var_renames:
                        var_renames[var_name] = "var{}".format(len(var_renames) + 1)
                    var_nodes.append([child_node, var_renames[var_name]])
                queue.append(child_node)
        return var_nodes

    def transform(self, id_nodes, code_text):
        id_nodes = sorted(id_nodes, reverse=True, key=lambda x: x[0].start_byte)
        for var_node, var_rename in id_nodes:
            code_text = code_text[:var_node.start_byte] + var_rename + code_text[var_node.end_byte:]
        return code_text

    def rename_identifiers(self, code_snippet):
        tree = self.parse(code_snippet)
        identifier_nodes = self.get_identifier_nodes(tree, code_snippet)
        return self.transform(identifier_nodes, code_snippet)