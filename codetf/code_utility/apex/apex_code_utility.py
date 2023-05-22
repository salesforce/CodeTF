from codetf.code_utility.language_specific_utility import LanguageSpecificUtility

class ApexCodeUtility(LanguageSpecificUtility):
    def __init__(self):
        super(ApexCodeUtility, self).__init__('apex')
        self.var_node_types = {'identifier'}
        self.var_filter_types = {'class_declaration', 'method_declaration', 'method_invocation'}
        self.comment_node_types = {'comment', 'line_comment','block_comment'}

    # Get only variable node from type "identifier"
    def get_identifier_nodes(self, tree, text):
        var_nodes = []
        var_renames = {}
        class_count, method_count = 1, 1
        queue = [tree.root_node]
        while queue:
            current_node = queue.pop(0)
            current_type = str(current_node.type)
            if current_type == "class_declaration":
                class_name_node = current_node.children[2]  # Assuming class name is the second child
                var_nodes.append([class_name_node, "class_{}".format(class_count)])
                class_count += 1
            elif current_type == "method_declaration":
                method_name_node = current_node.children[2]  # Assuming method name is the second child
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
    

    # def get_comment_nodes(self, tree, text):
    #     comment_nodes = []
    #     queue = [tree.root_node]
    #     while queue:
    #         current_node = queue.pop(0)
    #         current_type = str(current_node.type)
    #         print(current_type)
    #         if current_type in self.comment_node_types:
    #             comment_nodes.append(current_node)
    #         for child_node in current_node.children:
    #             queue.append(child_node)
    #     return comment_nodes


    def rename_identifiers(self, code_snippet):
        tree = self.parse(code_snippet)
        identifier_nodes = self.get_identifier_nodes(tree, code_snippet)
        identifier_nodes = sorted(identifier_nodes, reverse=True, key=lambda x: x[0].start_byte)
        for var_node, var_rename in identifier_nodes:
            code_snippet = code_snippet[:var_node.start_byte] + var_rename + code_snippet[var_node.end_byte:]
        return code_snippet
    
    # def remove_comments(self, code_snippet):
    #     tree = self.parse(code_snippet)
    #     comment_nodes = self.get_comment_nodes(tree, code_snippet)
    #     comment_nodes = sorted(comment_nodes, reverse=True, key=lambda x: x.start_byte)
    #     for comment_node in comment_nodes:
    #         code_snippet = code_snippet[:comment_node.start_byte] + code_snippet[comment_node.end_byte:]
    #     return code_snippet
    
    def extract_attributes(self, tree, text):
        attributes = {
            'class_names': [],
            'method_names': [],
            'comments': [],
            'variable_names': []
        }

        queue = [tree.root_node]
        while queue:
            current_node = queue.pop(0)
            current_type = str(current_node.type)

            if current_type == "class_declaration":
                class_name_node = current_node.children[2]  # Assuming class name is the second child
                class_name = text[class_name_node.start_byte: class_name_node.end_byte]
                attributes['class_names'].append(class_name)
            elif current_type == "method_declaration":
                method_name_node = current_node.children[2]  # Assuming method name is the second child
                method_name = text[method_name_node.start_byte: method_name_node.end_byte]
                attributes['method_names'].append(method_name)
            elif current_type == "comment":
                comment_text = text[current_node.start_byte: current_node.end_byte]
                attributes['comments'].append(comment_text)
            elif current_type == "identifier" and current_node.parent.type not in {'class_declaration', 'method_declaration'}:
                var_name = text[current_node.start_byte: current_node.end_byte]
                if var_name not in attributes['variable_names']:
                    attributes['variable_names'].append(var_name)

            queue.extend(current_node.children)

        return attributes

    def get_code_attributes(self, code_snippet):
        tree = self.parse(code_snippet)
        return self.extract_attributes(tree, code_snippet)
