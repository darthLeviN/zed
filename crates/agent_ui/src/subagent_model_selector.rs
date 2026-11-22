use std::sync::Arc;

use agent_settings::AgentSettings;
use fs::Fs;
use gpui::{DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Subscription};
use language_model::{LanguageModel, LanguageModelProviderId, LanguageModelRegistry};
use settings::{
    LanguageModelProviderSetting, LanguageModelSelection, Settings, SettingsContent, SettingsStore,
    update_settings_file,
};
use ui::{Button, ListSeparator, prelude::*};
use workspace::{ModalView, Workspace};

use crate::{
    AgentPanel, SelectSubagentModel,
    language_model_selector::{LanguageModelSelector, language_model_selector},
};

pub(crate) struct SubagentModelSelector {
    fs: Arc<dyn Fs>,
    picker: Entity<LanguageModelSelector>,
    parent_model: Option<LanguageModel>,
    _subscriptions: Vec<Subscription>,
}

impl SubagentModelSelector {
    pub(crate) fn register(
        workspace: &mut Workspace,
        _window: Option<&mut Window>,
        _cx: &mut Context<Workspace>,
    ) {
        workspace.register_action(|workspace, _: &SelectSubagentModel, window, cx| {
            let Some(panel) = workspace.panel::<AgentPanel>(cx) else {
                return;
            };
            let panel = panel.read(cx);
            let is_native = panel
                .active_conversation_view()
                .map(|view| view.read(cx).agent_key().is_native())
                .unwrap_or_else(|| panel.selected_agent(cx).is_native());
            if !is_native {
                return;
            }
            let parent_model = panel
                .active_native_agent_thread(cx)
                .and_then(|thread| thread.read(cx).model().cloned());
            let fs = workspace.app_state().fs.clone();
            workspace.toggle_modal(window, cx, |window, cx| {
                Self::new(fs, parent_model, window, cx)
            });
        });
    }

    fn new(
        fs: Arc<dyn Fs>,
        parent_model: Option<LanguageModel>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let picker = cx.new(|cx| {
            language_model_selector(
                configured_model,
                {
                    let fs = fs.clone();
                    move |model, cx| {
                        update_settings_file(fs.clone(), cx, move |settings, cx| {
                            let selection = selection_for_model(
                                &model,
                                AgentSettings::get_global(cx).subagent_model.as_ref(),
                            );
                            set_subagent_model(settings, Some(selection));
                        });
                    }
                },
                {
                    let fs = fs.clone();
                    move |model, should_be_favorite, cx| {
                        crate::favorite_models::toggle_in_settings(
                            model,
                            should_be_favorite,
                            fs.clone(),
                            cx,
                        );
                    }
                },
                false,
                focus_handle,
                window,
                cx,
            )
        });
        let subscriptions = vec![
            cx.subscribe(&picker, |_, _, _: &DismissEvent, cx| {
                cx.emit(DismissEvent);
            }),
            cx.observe_global::<SettingsStore>(|_, cx| cx.notify()),
        ];
        Self {
            fs,
            picker,
            parent_model,
            _subscriptions: subscriptions,
        }
    }
}

fn configured_model(cx: &App) -> Option<LanguageModel> {
    let selection = AgentSettings::get_global(cx).subagent_model.as_ref()?;
    LanguageModelRegistry::read_global(cx)
        .provider(&LanguageModelProviderId::from(selection.provider.0.clone()))?
        .provided_models(cx)
        .into_iter()
        .find(|model| model.id().0 == selection.model.as_str())
}

pub(crate) fn button_label(cx: &App) -> SharedString {
    let selection = AgentSettings::get_global(cx).subagent_model.as_ref();
    let name = configured_model(cx).map(|model| model.name().0);
    format_button_label(selection, name.as_deref())
}

fn format_button_label(
    selection: Option<&LanguageModelSelection>,
    model_name: Option<&str>,
) -> SharedString {
    match selection {
        Some(selection) => format!("Subagents: {}", model_name.unwrap_or(&selection.model)).into(),
        None => "Subagents: Parent Model".into(),
    }
}

fn selection_for_model(
    model: &LanguageModel,
    configured: Option<&LanguageModelSelection>,
) -> LanguageModelSelection {
    if let Some(configured) = configured.filter(|selection| {
        selection.provider.0 == model.provider_id().0.as_ref()
            && selection.model == model.id().0.as_ref()
    }) {
        return configured.clone();
    }
    LanguageModelSelection {
        provider: LanguageModelProviderSetting(model.provider_id().0.to_string()),
        model: model.id().0.to_string(),
        enable_thinking: model.supports_thinking(),
        effort: model
            .default_effort_level()
            .map(|effort| effort.value.to_string()),
        speed: None,
    }
}

fn set_subagent_model(settings: &mut SettingsContent, selection: Option<LanguageModelSelection>) {
    let agent = settings.agent.get_or_insert_default();
    if let (Some(configured), Some(selection)) = (&agent.subagent_model, &selection) {
        if configured.provider == selection.provider && configured.model == selection.model {
            return;
        }
    }
    agent.subagent_model = selection;
}

impl ModalView for SubagentModelSelector {}
impl EventEmitter<DismissEvent> for SubagentModelSelector {}

impl Focusable for SubagentModelSelector {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.picker.focus_handle(cx)
    }
}

impl Render for SubagentModelSelector {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selection = AgentSettings::get_global(cx).subagent_model.as_ref();
        let current_model = if let Some(selection) = selection {
            let name = configured_model(cx)
                .map(|model| model.name().0.to_string())
                .unwrap_or_else(|| selection.model.clone());
            format!("Current: {name} ({})", selection.provider.0)
        } else {
            self.parent_model
                .as_ref()
                .map(|model| format!("Current: Use Parent Model ({})", model.name().0))
                .unwrap_or_else(|| "Current: Use Parent Model".to_string())
        };

        v_flex()
            .elevation_3(cx)
            .w(rems(34.))
            .key_context("SubagentModelSelector")
            .on_action(cx.listener(|_, _: &menu::Cancel, _, cx| cx.emit(DismissEvent)))
            .on_mouse_down_out(cx.listener(|_, _, _, cx| cx.emit(DismissEvent)))
            .child(
                v_flex()
                    .p_3()
                    .gap_1()
                    .child(Label::new("Subagent Model"))
                    .child(
                        Label::new(current_model)
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    )
                    .child(
                        Label::new("Applies to new native Zed subagents, not the parent thread.")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
            )
            .child(ListSeparator)
            .child(self.picker.clone())
            .child(ListSeparator)
            .child(h_flex().p_2().child(
                Button::new("use-parent-model", "Use Parent Model").on_click(cx.listener(
                    |this, _, _, cx| {
                        update_settings_file(this.fs.clone(), cx, |settings, _| {
                            set_subagent_model(settings, None);
                        });
                        cx.emit(DismissEvent);
                    },
                )),
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selection(provider: &str, model: &str) -> LanguageModelSelection {
        LanguageModelSelection {
            provider: LanguageModelProviderSetting(provider.to_string()),
            model: model.to_string(),
            enable_thinking: true,
            effort: Some("high".to_string()),
            speed: Some(language_model::Speed::Fast),
        }
    }

    fn model(provider: &str, model: &str) -> LanguageModel {
        LanguageModel::new(
            language_model::LanguageModelId::from(model.to_string()),
            language_model::LanguageModelName::from(model.to_string()),
            LanguageModelProviderId::from(provider.to_string()),
            language_model::LanguageModelProviderName::from(provider.to_string()),
            "test",
            100_000,
        )
    }

    #[test]
    fn composer_label_shows_parent_or_selected_model() {
        assert_eq!(
            format_button_label(None, None).as_ref(),
            "Subagents: Parent Model"
        );
        let configured = selection("provider", "model-id");
        assert_eq!(
            format_button_label(Some(&configured), Some("Model Name")).as_ref(),
            "Subagents: Model Name"
        );
        assert_eq!(
            format_button_label(Some(&configured), None).as_ref(),
            "Subagents: model-id"
        );
    }

    #[test]
    fn new_subagent_model_uses_supported_defaults() {
        let mut model = model("provider", "model");
        model.supports_thinking = true;
        model.supported_effort_levels = vec![language_model::LanguageModelEffortLevel {
            name: "Medium".into(),
            value: "medium".into(),
            is_default: true,
            supports_disabling_thinking: None,
        }]
        .into();
        let configured = selection("provider", "old-model");
        let selection = selection_for_model(&model, Some(&configured));

        assert_eq!(selection.provider.0, "provider");
        assert_eq!(selection.model, "model");
        assert!(selection.enable_thinking);
        assert_eq!(selection.effort.as_deref(), Some("medium"));
        assert_eq!(selection.speed, None);
    }

    #[test]
    fn matching_effective_selection_preserves_metadata() {
        let model = model("provider", "model");
        let configured = selection("provider", "model");
        assert_eq!(selection_for_model(&model, Some(&configured)), configured);
    }

    #[test]
    fn different_provider_does_not_reuse_metadata() {
        let model = model("new-provider", "model");
        let configured = selection("old-provider", "model");
        let selection = selection_for_model(&model, Some(&configured));

        assert_eq!(selection.provider.0, "new-provider");
        assert!(!selection.enable_thinking);
        assert_eq!(selection.effort, None);
        assert_eq!(selection.speed, None);
    }

    #[test]
    fn separate_subagent_model_preserves_parent_and_metadata() {
        let mut settings = SettingsContent::default();
        let parent = selection("parent-provider", "parent-model");
        settings.agent.get_or_insert_default().default_model = Some(parent.clone());
        let subagent = selection("subagent-provider", "subagent-model");

        set_subagent_model(&mut settings, Some(subagent.clone()));

        let agent = settings.agent.as_ref().expect("agent settings");
        assert_eq!(agent.default_model, Some(parent));
        assert_eq!(agent.subagent_model, Some(subagent));
    }

    #[test]
    fn reselecting_subagent_model_preserves_configured_metadata() {
        let mut settings = SettingsContent::default();
        let configured = selection("provider", "model");
        settings.agent.get_or_insert_default().subagent_model = Some(configured.clone());
        let mut defaults = configured.clone();
        defaults.enable_thinking = false;
        defaults.effort = None;
        defaults.speed = None;

        set_subagent_model(&mut settings, Some(defaults));

        assert_eq!(
            settings
                .agent
                .as_ref()
                .expect("agent settings")
                .subagent_model,
            Some(configured)
        );
    }

    #[test]
    fn use_parent_model_clears_only_subagent_override() {
        let mut settings = SettingsContent::default();
        let parent = selection("parent-provider", "parent-model");
        let inline = selection("inline-provider", "inline-model");
        let agent = settings.agent.get_or_insert_default();
        agent.default_model = Some(parent.clone());
        agent.inline_assistant_model = Some(inline.clone());
        agent.subagent_model = Some(selection("subagent-provider", "subagent-model"));

        set_subagent_model(&mut settings, None);

        let agent = settings.agent.as_ref().expect("agent settings");
        assert_eq!(agent.subagent_model, None);
        assert_eq!(agent.default_model, Some(parent));
        assert_eq!(agent.inline_assistant_model, Some(inline));
    }
}
